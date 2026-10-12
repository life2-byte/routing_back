"""
Weather-aware truck routing API (single-file backend).

Sections:
  1. Risk rules          4. Weather (Open-Meteo)
  2. Geo helpers         5. Plan builder + ranking
  3. Routing (OSRM)      6. HTTP views
"""
import json
import math
import time
from bisect import bisect_left
from datetime import datetime, timedelta, timezone

import requests
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
METEO_URL = "https://api.open-meteo.com/v1/forecast"

EARTH_MI = 3958.8
MILE_M = 1609.344
MAX_AHEAD_DAYS = 15          # Open-Meteo forecast horizon (today + 15 days)
HEAT_HOURS = 48              # heatmap slider: 0..48h -> 49 values per point
HEAT_MAX_SPACING_MI = 25
MAX_CHECKPOINTS = 120        # per route, protects the weather API
TOLERANCE_MI = 5             # miles difference treated as "equal" when ranking
CACHE_TTL = 1800


class RoutingError(Exception):
    pass


class WeatherError(Exception):
    pass


# ---------------------------------------------------------------------------
# 1. Risk rules  (0=Low, 1=Moderate, 2=High, 3=Severe, 4=No Travel)
#    Boundaries are lower-inclusive, so a value sitting exactly on a boundary
#    goes to the HIGHER level, except the top bands where the spec says
#    "0.50-1.00" / "2.0-3.0" are Severe and ">1.00" / ">3.0" is No Travel.
# ---------------------------------------------------------------------------
def wind_level(mph):
    if mph >= 55:
        return 4
    if mph >= 45:
        return 3
    if mph >= 35:
        return 2
    if mph >= 25:
        return 1
    return 0


def rain_level(in_hr):
    if in_hr > 1.00:
        return 4
    if in_hr >= 0.50:
        return 3
    if in_hr >= 0.25:
        return 2
    if in_hr >= 0.10:
        return 1
    return 0


def snow_level(in_hr):
    if in_hr > 3.0:
        return 4
    if in_hr >= 2.0:
        return 3
    if in_hr >= 1.0:
        return 2
    if in_hr >= 0.5:
        return 1
    return 0


def classify(wind, rain, snow, load_lbs):
    """Overall level = max(wind, rain, snow), then load rules can escalate it."""
    level = max(wind_level(wind), rain_level(rain), snow_level(snow))
    if wind >= 55:                              # any load
        level = 4
    elif wind >= 45 and load_lbs > 30000:
        level = 4
    elif wind >= 35 and load_lbs > 40000:
        level = max(level, 3)
    return level


# ---------------------------------------------------------------------------
# 2. Geo helpers  (coords are (lon, lat) tuples)
# ---------------------------------------------------------------------------
def haversine(a, b):
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = math.radians(b[0] - a[0])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_MI * math.asin(math.sqrt(h))


def cumulative(coords, target_total):
    """Cumulative miles along the line, rescaled so the end equals OSRM's distance."""
    raw = [0.0]
    for a, b in zip(coords, coords[1:]):
        raw.append(raw[-1] + haversine(a, b))
    k = target_total / raw[-1] if raw[-1] else 1.0
    return [x * k for x in raw]


def point_at(coords, cum, mile):
    m = min(max(mile, 0.0), cum[-1])
    i = bisect_left(cum, m)
    if i <= 0:
        return coords[0]
    if i >= len(cum):
        return coords[-1]
    seg = (cum[i] - cum[i - 1]) or 1e-9
    t = (m - cum[i - 1]) / seg
    a, b = coords[i - 1], coords[i]
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def simplify(coords, cum, min_gap):
    """Thin the polyline so the frontend map stays smooth."""
    out = [coords[0]]
    last = cum[0]
    for c, m in zip(coords[1:-1], cum[1:-1]):
        if m - last >= min_gap:
            out.append(c)
            last = m
    out.append(coords[-1])
    return [[round(x, 5), round(y, 5)] for x, y in out]


def pick_interval(total, requested):
    if requested == "auto":
        interval = 10 if total < 150 else 25 if total < 400 else 50
    else:
        interval = float(requested)
    return max(interval, round(total / MAX_CHECKPOINTS, 1))


def checkpoint_miles(total, interval):
    miles = [i * interval for i in range(int(total // interval) + 1)]
    if total - miles[-1] < 1:
        miles[-1] = total
    else:
        miles.append(total)
    return miles


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 3. Routing (OSRM public server, car profile, free, no key)
# ---------------------------------------------------------------------------
def _osrm(points, alternatives):
    path = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in points)
    params = {
        "alternatives": "3" if alternatives else "false",
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
    }
    last_err = None
    for _ in range(2):
        try:
            r = requests.get(f"{OSRM_URL}/{path}", params=params, timeout=30)
            try:
                data = r.json()
            except ValueError:
                data = {}
            if r.status_code == 200 and data.get("code") == "Ok":
                return data["routes"]
            last_err = data.get("message") or data.get("code") or f"status {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1)
    raise RoutingError(f"Routing service failed: {last_err}")


def _similar(a, b):
    ca, cb = a["geometry"]["coordinates"], b["geometry"]["coordinates"]
    ma, mb = ca[len(ca) // 2], cb[len(cb) // 2]
    return abs(a["distance"] - b["distance"]) < 0.01 * a["distance"] and haversine(ma, mb) < 2


def get_routes(o, d):
    """Return up to 3 distinct routes. If OSRM gives fewer, force extras via waypoints."""
    uniq = []
    for r in _osrm([o, d], True):
        if not any(_similar(r, u) for u in uniq):
            uniq.append(r)

    if len(uniq) < 3:
        coords = uniq[0]["geometry"]["coordinates"]
        mid = coords[len(coords) // 2]
        dx, dy = d[0] - o[0], d[1] - o[1]
        norm = math.hypot(dx, dy) or 1.0
        px, py = -dy / norm, dx / norm
        offset = min(max(norm * 0.15, 0.1), 1.0)
        for sign in (1, -1):
            if len(uniq) >= 3:
                break
            via = (mid[0] + px * offset * sign, mid[1] + py * offset * sign)
            try:
                for r in _osrm([o, via, d], False):
                    if not any(_similar(r, u) for u in uniq):
                        uniq.append(r)
            except RoutingError:
                continue
    return uniq[:3]


# ---------------------------------------------------------------------------
# 4. Weather (Open-Meteo, free, no key). One batched call set feeds both
#    the checkpoints and the heatmap. Times are GMT so ETAs map directly.
# ---------------------------------------------------------------------------
def _meteo_request(chunk, start_date, end_date):
    params = {
        "latitude": ",".join(str(p[0]) for p in chunk),
        "longitude": ",".join(str(p[1]) for p in chunk),
        "hourly": "wind_speed_10m,rain,showers,snowfall",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "GMT",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    last_err = "unknown error"
    for attempt in range(3):
        try:
            r = requests.get(METEO_URL, params=params, timeout=30)
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else [data]
        try:
            last_err = r.json().get("reason", f"status {r.status_code}")
        except ValueError:
            last_err = f"status {r.status_code}"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    raise WeatherError(f"Weather service failed: {last_err}")


def fetch_weather(points, start_date, end_date):
    """points: iterable of (lat, lon) rounded to 2 decimals.
    Returns {(lat, lon): {"wind": [...], "rain": [...], "snow": [...]}} (hourly from start_date 00:00 GMT)."""
    pts = list(dict.fromkeys(points))
    keys = {p: f"wx:{p[0]}:{p[1]}:{start_date}:{end_date}" for p in pts}
    cached = cache.get_many(list(keys.values()))

    result, missing = {}, []
    for p, k in keys.items():
        if k in cached:
            result[p] = cached[k]
        else:
            missing.append(p)

    for i in range(0, len(missing), 50):
        chunk = missing[i:i + 50]
        for p, item in zip(chunk, _meteo_request(chunk, start_date, end_date)):
            h = item["hourly"]
            wx = {
                "wind": [v or 0 for v in h["wind_speed_10m"]],
                "rain": [(a or 0) + (b or 0) for a, b in zip(h["rain"], h["showers"])],
                "snow": [v or 0 for v in h["snowfall"]],
            }
            result[p] = wx
            cache.set(keys[p], wx, CACHE_TTL)

    for p in pts:
        if p not in result:
            raise WeatherError("Weather data missing for some checkpoints.")
    return result


def wx_at(wx, idx):
    i = min(max(idx, 0), len(wx["wind"]) - 1)
    return wx["wind"][i], wx["rain"][i], wx["snow"][i]


def hour_index(t, start_midnight):
    return round((t - start_midnight).total_seconds() / 3600)


# ---------------------------------------------------------------------------
# 5. Plan builder + ranking
# ---------------------------------------------------------------------------
def rank_key(r):
    s = r["summary"]
    return (
        1 if s["no_travel_miles"] > 0 else 0,        # No Travel routes always last
        math.floor(s["severe_miles"] / TOLERANCE_MI),  # 1) fewest Severe miles
        math.floor(s["high_miles"] / TOLERANCE_MI),    # 2) fewest High miles
        round(s["avg_risk"], 1),                       # 3) lowest average risk
        r["duration_min"],                             # 4) shortest time
    )


def build_plan(req):
    o, d = req["origin"], req["destination"]
    dep, load = req["departure"], req["load_lbs"]

    raw_routes = get_routes((o["lon"], o["lat"]), (d["lon"], d["lat"]))

    # ---- geometry, checkpoints, heatmap sample points (no weather yet) ----
    prepared = []
    heat_seen = set()
    heat_points = []
    for r in raw_routes:
        coords = [tuple(c) for c in r["geometry"]["coordinates"]]
        total = r["distance"] / MILE_M
        if total < 0.5:
            raise RoutingError("Origin and destination are too close.")
        cum = cumulative(coords, total)
        duration_min = r["duration"] / 60.0
        interval = pick_interval(total, req["interval_miles"])

        cps = []
        for m in checkpoint_miles(total, interval):
            lon, lat = point_at(coords, cum, m)
            cps.append({
                "mile": m, "lon": lon, "lat": lat,
                "eta": dep + timedelta(minutes=duration_min * m / total),
            })

        spacing = min(HEAT_MAX_SPACING_MI, max(5.0, total / 40))
        miles, m = [], 0.0
        while m < total:
            miles.append(m)
            m += spacing
        miles.append(total)
        for m in miles:
            lon, lat = point_at(coords, cum, m)
            cell = (round(lat, 1), round(lon, 1))   # ~weather model grid size
            if cell not in heat_seen:
                heat_seen.add(cell)
                heat_points.append((lat, lon))

        prepared.append({
            "coords": coords, "cum": cum, "total": total,
            "duration_min": duration_min, "interval": interval, "cps": cps,
        })

    # ---- weather: one batched fetch window for everything ----
    now = datetime.now(timezone.utc)
    max_dur_h = max(p["duration_min"] for p in prepared) / 60.0
    horizon = dep + timedelta(hours=max(HEAT_HOURS, max_dur_h) + 1)
    start_date = dep.date()
    end_date = min(horizon.date(), (now + timedelta(days=MAX_AHEAD_DAYS)).date())
    start_midnight = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)

    wkey = lambda lat, lon: (round(lat, 2), round(lon, 2))
    needed = [wkey(c["lat"], c["lon"]) for p in prepared for c in p["cps"]]
    needed += [wkey(lat, lon) for lat, lon in heat_points]
    wx = fetch_weather(needed, start_date, end_date)

    # ---- per-route results ----
    routes = []
    for idx, p in enumerate(prepared):
        cps_out, by_level, weighted = [], [0.0] * 5, 0.0
        for i, c in enumerate(p["cps"]):
            wind, rain, snow = wx_at(wx[wkey(c["lat"], c["lon"])], hour_index(c["eta"], start_midnight))
            risk = classify(wind, rain, snow, load)
            nxt = p["cps"][i + 1]["mile"] if i + 1 < len(p["cps"]) else c["mile"]
            seg = nxt - c["mile"]
            by_level[risk] += seg
            weighted += risk * seg
            cps_out.append({
                "mile": round(c["mile"], 1),
                "lat": round(c["lat"], 5),
                "lon": round(c["lon"], 5),
                "eta": iso(c["eta"]),
                "wind_mph": round(wind, 1),
                "rain_in_hr": round(rain, 2),
                "snow_in_hr": round(snow, 2),
                "risk": risk,
                "segment_miles": round(seg, 1),
            })

        summary = {
            "severe_miles": round(by_level[3], 1),
            "high_miles": round(by_level[2], 1),
            "no_travel_miles": round(by_level[4], 1),
            "avg_risk": round(weighted / p["total"], 2),
            "max_risk": max(c["risk"] for c in cps_out),
            "miles_by_level": [round(v, 1) for v in by_level],
        }
        warnings = []
        if summary["no_travel_miles"] > 0:
            warnings.append("Contains No Travel conditions")

        routes.append({
            "id": f"r{idx + 1}",
            "name": f"Route {idx + 1}",
            "distance_mi": round(p["total"], 1),
            "duration_min": round(p["duration_min"]),
            "geometry": simplify(p["coords"], p["cum"], max(0.15, p["total"] / 1500)),
            "summary": summary,
            "checkpoints": cps_out,
            "warnings": warnings,
            "recommended": False,
        })

    best = min(routes, key=rank_key)
    best["recommended"] = True

    # ---- heatmap: 49 hourly risk values per point ----
    heat_out = []
    for lat, lon in heat_points:
        w = wx[wkey(lat, lon)]
        risk = []
        for h in range(HEAT_HOURS + 1):
            wind, rain, snow = wx_at(w, hour_index(dep + timedelta(hours=h), start_midnight))
            risk.append(classify(wind, rain, snow, load))
        heat_out.append({"lat": round(lat, 5), "lon": round(lon, 5), "risk": risk})

    return {
        "request": {
            "origin": o,
            "destination": d,
            "departure": iso(dep),
            "load_lbs": load,
            "interval_miles": prepared[0]["interval"],
        },
        "recommended_route_id": best["id"],
        "routes": routes,
        "heatmap": {"start": iso(dep), "hours": HEAT_HOURS + 1, "points": heat_out},
    }


# ---------------------------------------------------------------------------
# 6. HTTP views
# ---------------------------------------------------------------------------
def validate(p):
    def place(key):
        v = p.get(key) or {}
        try:
            lat, lon = float(v["lat"]), float(v["lon"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{key}: lat and lon are required.")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{key}: coordinates out of range.")
        return {"name": str(v.get("name", "")), "lat": lat, "lon": lon}

    origin, destination = place("origin"), place("destination")

    try:
        departure = datetime.fromisoformat(str(p["departure"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        raise ValueError("departure must be an ISO datetime.")
    if departure.tzinfo is None:
        departure = departure.replace(tzinfo=timezone.utc)
    departure = departure.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    if departure < now - timedelta(hours=1):
        raise ValueError("Departure cannot be in the past.")
    if departure > now + timedelta(days=MAX_AHEAD_DAYS):
        raise ValueError("Weather forecast is only available up to 16 days ahead.")

    try:
        load = float(p["load_lbs"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("load_lbs must be a number.")
    if load <= 0 or load > 200000:
        raise ValueError("load_lbs must be between 1 and 200,000.")

    interval = p.get("interval_miles", "auto")
    if interval != "auto":
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            raise ValueError("interval_miles must be 'auto', 10, 25 or 50.")
        if interval not in (10.0, 25.0, 50.0):
            raise ValueError("interval_miles must be 'auto', 10, 25 or 50.")

    return {
        "origin": origin, "destination": destination,
        "departure": departure, "load_lbs": load, "interval_miles": interval,
    }


def _cors(resp):
    resp["Access-Control-Allow-Origin"] = "*"
    resp["Access-Control-Allow-Headers"] = "Content-Type"
    resp["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def _err(msg, status):
    return _cors(JsonResponse({"error": msg}, status=status))


@csrf_exempt
def plan(request):
    if request.method == "OPTIONS":
        return _cors(HttpResponse(status=204))
    if request.method != "POST":
        return _err("Use POST.", 405)

    try:
        payload = json.loads(request.body or b"{}")
    except ValueError:
        return _err("Invalid JSON body.", 400)

    try:
        req = validate(payload)
    except ValueError as e:
        return _err(str(e), 400)

    try:
        return _cors(JsonResponse(build_plan(req)))
    except (RoutingError, WeatherError) as e:
        return _err(str(e), 502)
    except requests.RequestException as e:
        return _err(f"Upstream API error: {e}", 502)


def health(request):
    return _cors(JsonResponse({"status": "ok"}))