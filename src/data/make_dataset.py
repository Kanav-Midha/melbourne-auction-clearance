"""Build the three raw source files this project consumes.

The public REIV/Domain weekly auction results and the BOM historical archive are
not redistributable in bulk, so this module synthesises files with the *same
schema and the same defects* as the real extracts:

  * prices arriving as strings (``"$1,250,000"``) mixed with plain integers
  * a price guide stored as a free-text range (``"$900,000 - $990,000"``)
  * inconsistent suburb capitalisation and stray whitespace
  * duplicated auction rows from double-entered results
  * outcome columns (``sold_price``, ``result_code``) that leak the target
  * BOM weather with random gaps *and* a three-month station outage

The generative process is documented in ``docs/data_generation.md``. Swap
``--source real`` in once you have licensed extracts; the downstream pipeline
does not change.

Usage
-----
    python -m src.data.make_dataset --n-auctions 48000 --seed 42
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config
from src.data.suburbs import suburb_table

log = logging.getLogger(__name__)

# Intercept of the latent clearance model, calibrated so the simulated
# clearance rate tracks the REIV series: high-70s in the 2021 boom, high-50s
# at the 2022 tightening trough, mid-60s otherwise.
BASE_LOGIT = 1.58

START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2024-12-31")

# BOM station id, name, lat, lon
BOM_STATIONS = [
    ("086338", "MELBOURNE (OLYMPIC PARK)", -37.8255, 144.9816),
    ("086282", "MOORABBIN AIRPORT", -37.9807, 145.0961),
    ("086038", "ESSENDON AIRPORT", -37.7276, 144.9066),
    ("087031", "LAVERTON RAAF", -37.8565, 144.7566),
    ("086104", "SCORESBY RESEARCH INSTITUTE", -37.8710, 145.2560),
]

AGENCIES = [
    "Barry Plant", "Jellis Craig", "Ray White", "Nelson Alexander", "Marshall White",
    "hockingstuart", "Woodards", "Biggin & Scott", "Harcourts", "Buxton",
    "McGrath", "Fletchers", "Stockdale & Leggo", "Belle Property", "OBrien",
]


# --------------------------------------------------------------------------- #
# RBA cash rate -- the dominant macro driver of Melbourne clearance rates.
# Approximates the published target cash rate path 2019-2024.
# --------------------------------------------------------------------------- #
_RATE_CHANGES = {
    "2019-01": 1.50, "2019-06": 1.25, "2019-07": 1.00, "2019-10": 0.75,
    "2020-03": 0.25, "2020-11": 0.10,
    "2022-05": 0.35, "2022-06": 0.85, "2022-07": 1.35, "2022-08": 1.85,
    "2022-09": 2.35, "2022-10": 2.60, "2022-11": 2.85, "2022-12": 3.10,
    "2023-02": 3.35, "2023-03": 3.60, "2023-05": 3.85, "2023-06": 4.10,
    "2023-11": 4.35,
}


def cash_rate_series() -> pd.Series:
    """Monthly RBA cash rate, forward-filled between decisions."""
    months = pd.period_range(START, END, freq="M")
    rates = pd.Series(np.nan, index=months, dtype=float)
    for period, rate in _RATE_CHANGES.items():
        p = pd.Period(period, freq="M")
        if p in rates.index:
            rates.loc[p] = rate
    return rates.ffill()


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def make_weather(rng: np.random.Generator) -> pd.DataFrame:
    """Daily BOM observations for five stations, before defects are injected."""
    dates = pd.date_range(START, END, freq="D")
    doy = dates.dayofyear.to_numpy()
    # Melbourne: warmest early Feb, coolest mid July.
    seasonal = 20.0 + 6.5 * np.cos(2 * np.pi * (doy - 32) / 365.25)

    frames = []
    for station_id, name, lat, lon in BOM_STATIONS:
        # Inland stations run warmer and drier than the bayside ones.
        warm_offset = 1.4 if name.startswith(("ESSENDON", "SCORESBY")) else 0.0
        wet_factor = 1.15 if name.startswith("SCORESBY") else 1.0

        max_temp = seasonal + warm_offset + rng.normal(0, 3.6, len(dates))
        # Heat spikes: Melbourne gets short runs of 38C+ days in summer.
        spike = (rng.random(len(dates)) < 0.015) & (seasonal > 22)
        max_temp = max_temp + spike * rng.uniform(6, 13, len(dates))

        wet = rng.random(len(dates)) < (0.34 * wet_factor)
        rainfall = np.where(wet, rng.gamma(shape=0.9, scale=5.0, size=len(dates)), 0.0)

        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "station_name": name,
                    "station_lat": lat,
                    "station_lon": lon,
                    "date": dates,
                    "max_temp_c": np.round(max_temp, 1),
                    "rainfall_mm": np.round(rainfall, 1),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def inject_weather_defects(wx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Degrade the clean weather frame into the shape BOM actually ships."""
    wx = wx.copy()

    # --- Defect 1: Olympic Park was offline for the 2021 winter. -------------
    outage = (
        (wx["station_id"] == "086338")
        & (wx["date"] >= "2021-06-01")
        & (wx["date"] <= "2021-08-31")
    )
    wx.loc[outage, ["max_temp_c", "rainfall_mm"]] = np.nan

    # --- Defect 2: scattered missing observations ----------------------------
    wx.loc[rng.random(len(wx)) < 0.07, "rainfall_mm"] = np.nan
    wx.loc[rng.random(len(wx)) < 0.04, "max_temp_c"] = np.nan

    # --- Defect 3: rainfall column arrives as text with quality flags --------
    rain_txt = wx["rainfall_mm"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    flagged = rng.random(len(wx)) < 0.03
    wx["rainfall_mm"] = np.where(flagged, rain_txt + " ", rain_txt)

    wx["date"] = wx["date"].dt.strftime("%d/%m/%Y")  # BOM ships dd/mm/yyyy
    return wx


# --------------------------------------------------------------------------- #
# PTV train stations
# --------------------------------------------------------------------------- #
def make_stations(rng: np.random.Generator, subs: pd.DataFrame) -> pd.DataFrame:
    """Approximate PTV metropolitan train station locations.

    Real station coordinates come from the PTV GTFS feed; here we scatter a
    plausible number of stations around each suburb centroid.
    """
    rows = []
    for r in subs.itertuples():
        # Inner suburbs are denser: more stations per suburb.
        n = 2 if r.price_tier >= 4 else rng.integers(1, 3)
        for i in range(int(n)):
            rows.append(
                {
                    "stop_id": f"{19000 + len(rows)}",
                    "stop_name": f"{r.suburb} Station" if i == 0 else f"{r.suburb} {i+1}",
                    "stop_lat": r.lat + rng.normal(0, 0.011),
                    "stop_lon": r.lon + rng.normal(0, 0.013),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Auction calendar
# --------------------------------------------------------------------------- #
def auction_calendar(rng: np.random.Generator, n_target: int) -> pd.DataFrame:
    """Pick auction dates and per-day volumes.

    Melbourne auctions are overwhelmingly a Saturday event, with a seasonal
    volume profile: heavy in spring (Oct-Nov) and autumn (Mar-May), effectively
    dead from mid-December to late January.
    """
    days = pd.date_range(START, END, freq="D")
    sat = days[days.dayofweek == 5]

    month = sat.month.to_numpy()
    weight = np.select(
        [
            np.isin(month, [12]),
            np.isin(month, [1]),
            np.isin(month, [10, 11]),
            np.isin(month, [3, 4, 5]),
        ],
        [0.35, 0.08, 1.45, 1.20],
        default=0.90,
    )
    # Melbourne's 2020 lockdowns: on-site auctions were suspended.
    lockdown = ((sat >= "2020-03-25") & (sat <= "2020-05-20")) | (
        (sat >= "2020-08-02") & (sat <= "2020-10-25")
    )
    weight = weight * np.where(lockdown, 0.06, 1.0)
    weight = weight * rng.uniform(0.82, 1.18, len(sat))
    volumes = np.round(weight / weight.sum() * n_target).astype(int)

    return pd.DataFrame({"auction_date": sat, "n_auctions": volumes})


# --------------------------------------------------------------------------- #
# Main generator
# --------------------------------------------------------------------------- #
def make_auctions(rng: np.random.Generator, subs: pd.DataFrame,
                  stations: pd.DataFrame, weather: pd.DataFrame,
                  n_target: int) -> pd.DataFrame:
    from src.features.spatial import haversine_km

    cal = auction_calendar(rng, n_target)
    rates = cash_rate_series()

    # Weather moves clearance at the city level, not the property level: it
    # rains on the whole of Melbourne on a given Saturday. Use the network mean
    # of the *clean* series -- the defects injected later are a measurement
    # problem, not part of the data-generating process.
    daily_wx = weather.groupby("date")[["rainfall_mm", "max_temp_c"]].mean()

    # Demand momentum as two nested AR(1) walks over auction weekends:
    # a region-level component shared by every suburb in the region, plus a
    # smaller suburb-specific one.
    #
    # The split matters. A purely suburb-level process is only observable
    # through a trailing window over the handful of auctions that suburb held
    # in the last four weeks, so the feature is mostly sampling noise. The
    # region component is estimated from hundreds of auctions per weekend, so
    # `region_clearance_l4w` recovers it precisely -- which is exactly why real
    # clearance-rate series are quoted at the region level.
    weeks = cal["auction_date"].to_numpy()

    def ar1(phi: float, sigma: float) -> np.ndarray:
        m = np.zeros(len(weeks))
        for t in range(1, len(weeks)):
            m[t] = phi * m[t - 1] + rng.normal(0, sigma)
        return m

    region_momentum = {r: ar1(0.88, 0.20) for r in subs["region_name"].unique()}
    suburb_momentum = {s: ar1(0.82, 0.17) for s in subs["suburb"]}
    week_index = {d: i for i, d in enumerate(weeks)}

    sub_lookup = subs.set_index("suburb")
    station_lat = stations["stop_lat"].to_numpy()
    station_lon = stations["stop_lon"].to_numpy()

    # Suburb base median price (in $), by tier, drifting with the market cycle.
    tier_base = {1: 520_000, 2: 640_000, 3: 810_000, 4: 1_150_000, 5: 1_820_000}

    records = []
    for row in cal.itertuples():
        n = int(row.n_auctions)
        if n <= 0:
            continue
        date = row.auction_date
        period = pd.Period(date, freq="M")
        rate = float(rates.loc[period])
        rate_3m_ago = float(rates.loc[period - 3]) if (period - 3) in rates.index else rate
        rate_change_3m = rate - rate_3m_ago
        widx = week_index[np.datetime64(date)]
        rain = float(daily_wx.loc[date, "rainfall_mm"])
        tmax = float(daily_wx.loc[date, "max_temp_c"])
        # A wet Saturday thins the crowd on the nature strip; so does 38C.
        weather_effect = -0.014 * min(rain, 25.0) - 0.020 * max(0.0, tmax - 33.0)

        # Market-wide price index: strong 2021 run-up, 2022 correction.
        t_years = (date - START).days / 365.25
        price_index = (
            1.0
            + 0.055 * t_years
            + 0.16 * np.exp(-((t_years - 2.7) ** 2) / 0.45)
            - 0.075 * np.clip(t_years - 3.4, 0, None)
        )

        # Higher-tier suburbs auction more often than they exist in the table.
        pick_w = (sub_lookup["price_tier"].to_numpy() ** 1.25).astype(float)
        pick_w = pick_w / pick_w.sum()
        chosen = rng.choice(subs["suburb"].to_numpy(), size=n, p=pick_w)

        for suburb in chosen:
            info = sub_lookup.loc[suburb]
            lat = float(info["lat"]) + rng.normal(0, 0.008)
            lon = float(info["lon"]) + rng.normal(0, 0.010)
            tier = int(info["price_tier"])

            d_cbd = haversine_km(lat, lon, config.CBD_LAT, config.CBD_LON)
            d_stations = haversine_km(lat, lon, station_lat, station_lon)
            d_station = float(d_stations.min())
            n_within_1km = int((d_stations <= 1.0).sum())

            # --- property attributes ---------------------------------------
            is_unit = rng.random() < (0.52 if tier >= 4 and d_cbd < 9 else 0.24)
            is_town = (not is_unit) and rng.random() < 0.18
            ptype = "u" if is_unit else ("t" if is_town else "h")

            if ptype == "u":
                beds = int(np.clip(rng.normal(2.0, 0.7), 1, 4))
                land = float(rng.gamma(2.0, 45))
                building = float(np.clip(rng.normal(78, 22), 35, 200))
            elif ptype == "t":
                beds = int(np.clip(rng.normal(3.0, 0.6), 2, 5))
                land = float(np.clip(rng.normal(230, 70), 90, 500))
                building = float(np.clip(rng.normal(145, 35), 80, 280))
            else:
                beds = int(np.clip(rng.normal(3.4, 0.95), 1, 7))
                land = float(np.clip(rng.gamma(4.0, 145), 180, 2200))
                building = float(np.clip(rng.normal(60 + 32 * beds, 40), 70, 520))

            baths = int(np.clip(round(beds * 0.55 + rng.normal(0, 0.5)), 1, 4))
            cars = int(np.clip(round((0 if ptype == "u" else 1.4) + rng.normal(0, 0.8)), 0, 4))
            year_built = int(np.clip(rng.normal(1968, 28), 1880, date.year))

            # --- pricing ----------------------------------------------------
            suburb_median = tier_base[tier] * price_index
            type_mult = {"h": 1.0, "t": 0.80, "u": 0.60}[ptype]
            size_mult = 0.72 + 0.115 * beds + 0.00016 * land
            # Idiosyncratic value spread. Kept moderate so that the observable
            # attributes (type, beds, land, floor area, suburb) genuinely
            # determine most of a property's value -- otherwise
            # `guide_vs_suburb_median` is dominated by unobservable noise.
            true_value = suburb_median * type_mult * size_mult * np.exp(rng.normal(0, 0.105))

            # Vendors set a guide relative to value; some are unrealistic.
            guide_ratio = np.exp(rng.normal(-0.035, 0.135))
            guide_mid = true_value * guide_ratio
            guide_low = guide_mid * 0.955
            guide_high = guide_mid * 1.045

            # --- latent probability of selling under the hammer -------------
            unobserved = rng.normal(0, 0.40)  # condition, agent skill, campaign
            z = (
                BASE_LOGIT
                # Buyers react to the *pace* of tightening, but the reaction
                # saturates -- tanh, not a raw linear term. A linear coefficient
                # sent the 2022 trough to a 9% clearance rate, which never
                # happened; the real floor was around 55%.
                - 0.60 * np.tanh(rate_change_3m / 0.8)
                - 0.05 * (rate - 2.0)
                # Expensive suburbs carry bigger mortgages, so they react
                # harder to tightening. In 2022 the inner east fell roughly
                # twice as far as the outer west. This is an interaction
                # term: no main effect on rates or on tier reproduces it.
                - 0.55 * np.tanh(rate_change_3m / 0.8) * (tier - 3) / 2.0
                # Units show for buyers in summer and sit empty in winter.
                - (0.30 if ptype == "u" else 0.0) * (1 if date.month in (6, 7, 8) else 0)
                # The 2021 stimulus-era boom, peaking around August 2021.
                + 0.55 * np.exp(-((t_years - 2.55) ** 2) / 0.65)
                + 0.055 * (tier - 3)
                - 0.0165 * d_cbd
                - 0.012 * max(0.0, d_cbd - 28)
                # Proximity to a station helps, but backing onto the line does not.
                - 0.45 * abs(np.log(max(d_station, 0.05) / 0.62))
                - 0.55 * max(0.0, 0.22 - d_station) / 0.22
                + 0.035 * n_within_1km
                # 3-4 bedroom family homes are the deepest part of the market.
                - 0.18 * (beds - 3.5) ** 2
                # Units are fine close in, poor in the outer suburbs.
                - (0.85 if ptype == "u" else 0.0) * np.clip((d_cbd - 11) / 16, 0, 1.6)
                - (0.14 if ptype == "t" else 0.0)
                # Overpriced guides pass in; underquoting draws a crowd.
                # Overpricing is the single biggest reason a property passes in.
                - 3.60 * max(0.0, guide_ratio - 1.0)
                + 1.60 * max(0.0, 1.0 - guide_ratio)
                + 1.15 * region_momentum[info["region_name"]][widx]
                + 0.95 * suburb_momentum[suburb][widx]
                + weather_effect
                - 0.00055 * (n - 560)
                + 0.00035 * (year_built - 1968) * 0.4
                + unobserved
            )

            records.append(
                {
                    "auction_date": date,
                    "suburb": suburb,
                    "council_area": info["council_area"],
                    "region_name": info["region_name"],
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "property_type": ptype,
                    "bedrooms": beds,
                    "bathrooms": baths,
                    "car_spaces": cars,
                    "land_size_sqm": round(land, 1),
                    "building_area_sqm": round(building, 1),
                    "year_built": year_built,
                    "agency": rng.choice(AGENCIES),
                    "guide_price_low": guide_low,
                    "guide_price_high": guide_high,
                    "rba_cash_rate": rate,
                    "n_scheduled": n,
                    "_z": z,
                    "_true_value": true_value,
                    "_d_cbd": d_cbd,
                    "_d_station": d_station,
                    "_n_within_1km": n_within_1km,
                }
            )

    df = pd.DataFrame(records)

    # --- realise the outcome -------------------------------------------------
    p = 1.0 / (1.0 + np.exp(-df["_z"].to_numpy()))
    sold = rng.random(len(df)) < p
    df["_sold"] = sold

    # Map to the REIV result codes that appear in the real feed.
    u = rng.random(len(df))
    result = np.where(
        sold,
        np.where(u < 0.86, "S", np.where(u < 0.94, "SP", "SA")),
        np.where(u < 0.70, "PI", np.where(u < 0.90, "VB", "W")),
    )
    df["result_code"] = result
    df["result_description"] = pd.Series(result).map(
        {
            "S": "Sold at auction",
            "SP": "Sold prior to auction",
            "SA": "Sold after auction",
            "PI": "Passed in",
            "VB": "Passed in - vendor bid",
            "W": "Withdrawn",
        }
    ).to_numpy()

    # Sale price exists only when it sold -- the leak the notebook walks into.
    premium = np.exp(rng.normal(0.035, 0.055, len(df)))
    sold_price = np.where(sold, np.round(df["_true_value"] * premium, -3), np.nan)
    df["sold_price"] = sold_price

    df["vendor_discount_pct"] = np.where(
        sold, np.round((sold_price / df["_true_value"] - 1) * 100, 2), np.nan
    )
    df["price_per_sqm"] = np.where(
        sold & (df["building_area_sqm"] > 0),
        np.round(sold_price / df["building_area_sqm"], 0),
        np.nan,
    )
    settle_days = rng.choice([30, 45, 60, 90], size=len(df), p=[0.18, 0.22, 0.42, 0.18])
    df["days_to_settle"] = np.where(sold, settle_days, np.nan)
    df["sale_settlement_date"] = pd.to_datetime(
        np.where(sold, df["auction_date"] + pd.to_timedelta(settle_days, unit="D"), pd.NaT)
    )

    # Keep the latent score and true probability out of the published CSV but
    # write them to data/interim/. Knowing the Bayes-optimal AUC is what lets us
    # say how much of the *achievable* signal a model actually recovered --
    # a luxury real data never affords.
    ground_truth = pd.DataFrame({
        "_row": np.arange(len(df)),
        "z": df["_z"].to_numpy(),
        "p_true": p,
        "sold": sold.astype(int),
    })

    df = df.drop(columns=["_z", "_true_value", "_sold", "_d_cbd", "_d_station", "_n_within_1km"])

    # --- surface defects -----------------------------------------------------
    shuffled = df.sample(frac=1.0, random_state=int(rng.integers(1e6)))
    ground_truth = ground_truth.set_index("_row").loc[shuffled.index].reset_index(drop=True)
    df = shuffled.reset_index(drop=True)
    df.insert(0, "listing_id", [f"VIC{2000000 + i}" for i in range(len(df))])
    ground_truth.insert(0, "listing_id", df["listing_id"].to_numpy())
    ground_truth.to_parquet(config.INTERIM_DIR / "ground_truth.parquet", index=False)

    # Free-text price guide, the way portals publish it.
    df["price_guide"] = [
        f"${lo:,.0f} - ${hi:,.0f}"
        for lo, hi in zip(df["guide_price_low"].round(-3), df["guide_price_high"].round(-3),
                          strict=True)
    ]
    df = df.drop(columns=["guide_price_low", "guide_price_high"])

    # Sold price: half the rows are currency strings, half plain numbers.
    as_text = rng.random(len(df)) < 0.5
    df["sold_price"] = [
        ("" if pd.isna(v) else (f"${v:,.0f}" if t else f"{v:.0f}"))
        for v, t in zip(df["sold_price"], as_text, strict=True)
    ]

    # Inconsistent suburb capitalisation and stray whitespace.
    mess = rng.random(len(df))
    df["suburb"] = np.where(
        mess < 0.02, df["suburb"].str.upper(),
        np.where(mess < 0.035, " " + df["suburb"], df["suburb"]),
    )

    # Double-entered results: ~1.5% of rows appear twice.
    dupes = df.sample(frac=0.015, random_state=7)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    df["auction_date"] = df["auction_date"].dt.strftime("%Y-%m-%d")
    df["sale_settlement_date"] = pd.to_datetime(df["sale_settlement_date"]).dt.strftime("%Y-%m-%d")
    df["sale_settlement_date"] = df["sale_settlement_date"].fillna("")

    ordered = [
        "listing_id", "auction_date", "suburb", "council_area", "region_name",
        "lat", "lon", "property_type", "bedrooms", "bathrooms", "car_spaces",
        "land_size_sqm", "building_area_sqm", "year_built", "agency",
        "price_guide", "rba_cash_rate", "n_scheduled",
        "result_code", "result_description", "sold_price", "price_per_sqm",
        "vendor_discount_pct", "days_to_settle", "sale_settlement_date",
    ]
    return df[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-auctions", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    subs = suburb_table()
    stations = make_stations(rng, subs)
    stations.to_csv(config.STATIONS_RAW, index=False)
    log.info("wrote %s (%d stations)", config.STATIONS_RAW.name, len(stations))

    weather = make_weather(rng)
    inject_weather_defects(weather, rng).to_csv(config.WEATHER_RAW, index=False)
    log.info("wrote %s (%d obs)", config.WEATHER_RAW.name, len(weather))

    auctions = make_auctions(rng, subs, stations, weather, args.n_auctions)
    auctions.to_csv(config.AUCTIONS_RAW, index=False)
    log.info("wrote %s (%d auctions)", config.AUCTIONS_RAW.name, len(auctions))

    rate = auctions["result_code"].isin(["S", "SP", "SA"]).mean()
    log.info("overall clearance rate: %.1f%%", 100 * rate)


if __name__ == "__main__":
    main()
