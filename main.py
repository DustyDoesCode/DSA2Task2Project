#Dustin Snelgrove
#StudentID: 011633071
#C950 Task 2 Project

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterable
from datetime import datetime, timedelta
import csv

"""
WGUPS Task 2 — Python Implementation
---------------------------------------------------
HashTable with separate chaining used for:
  • packages keyed by package ID
  • address → index lookups for the distance matrix
Other temporary collections are lists. No third‑party libraries.
"""

# =============================
# Time utilities
# =============================

BASE_DAY = datetime(2025, 1, 1, 8, 0, 0)  # 8:00 AM arbitrary date
SPEED_MPH = 18.0
CAPACITY = 16


def _to_minutes(dt: datetime) -> int:
    return int((dt - BASE_DAY).total_seconds() // 60)


def parse_clock(s: str) -> int:
    s = s.strip().upper()
    if s in {"EOD", "END OF DAY"}:
        # Use 5:00 PM as an EOD sentinel
        return _to_minutes(datetime(2025, 1, 1, 17, 0, 0))
    s = s.replace(" ", "")
    ampm = s[-2:]
    hh, mm = s[:-2].split(":")
    h, m = int(hh), int(mm)
    if ampm == "PM" and h != 12:
        h += 12
    if ampm == "AM" and h == 12:
        h = 0
    return _to_minutes(datetime(2025, 1, 1, h, m))

START_MIN = 0
TEN_TWENTY = parse_clock("10:20 AM")
EOD_MIN = parse_clock("EOD")


# =============================
# Custom Hash Table (separate chaining)
# =============================

class HashTable:
    class _Entry:
        __slots__ = ("key", "value")
        def __init__(self, key, value):
            self.key = key
            self.value = value

    def __init__(self, capacity: int = 53):
        if capacity < 8:
            capacity = 8
        self._buckets: List[List[HashTable._Entry]] = [[] for _ in range(capacity)]
        self._size = 0

    def _index(self, key) -> int:
        # Use built-in hash for dispersion
        return (hash(key) & 0x7FFFFFFF) % len(self._buckets)

    def _load_factor(self) -> float:
        return self._size / float(len(self._buckets))

    def _rehash(self, new_cap: int):
        old = self._buckets
        self._buckets = [[] for _ in range(new_cap)]
        self._size = 0
        for bucket in old:
            for e in bucket:
                self.insert(e.key, e.value)

    def insert(self, key, value) -> None:
        idx = self._index(key)
        bucket = self._buckets[idx]
        for e in bucket:
            if e.key == key:
                e.value = value
                return
        bucket.append(HashTable._Entry(key, value))
        self._size += 1
        if self._load_factor() > 0.75:
            self._rehash(len(self._buckets) * 2)

    def get(self, key):
        idx = self._index(key)
        bucket = self._buckets[idx]
        for e in bucket:
            if e.key == key:
                return e.value
        return None

    def update(self, key, value) -> None:
        self.insert(key, value)

    def remove(self, key) -> bool:
        idx = self._index(key)
        bucket = self._buckets[idx]
        for i, e in enumerate(bucket):
            if e.key == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def items(self) -> Iterable[Tuple[object, object]]:
        for bucket in self._buckets:
            for e in bucket:
                yield (e.key, e.value)

    def keys(self) -> Iterable[object]:
        for k, _ in self.items():
            yield k

    def values(self) -> Iterable[object]:
        for _, v in self.items():
            yield v

    def __len__(self) -> int:
        return self._size


# =============================
# Data models
# =============================

@dataclass
class Package:
    id: int
    address: str
    city: str
    state: str
    zip: str
    deadline_min: int
    weight: int
    notes: str = ""
    # dynamic
    index: Optional[int] = None
    status: str = "at hub"  # at hub | en route | delivered
    time_loaded: Optional[int] = None
    time_delivered: Optional[int] = None
    earliest_ready: int = START_MIN
    truck_only: Optional[int] = None
    group_id: Optional[str] = None
    truck_loaded: Optional[int] = None
    truck_delivered: Optional[int] = None


    def deadline_rank(self) -> int:
        # 9:00 AM < 10:30 AM < EOD
        if self.deadline_min <= parse_clock("9:00 AM"):
            return 0
        if self.deadline_min <= parse_clock("10:30 AM"):
            return 1
        return 2


@dataclass
class Truck:
    id: int
    depart_min: int
    location_idx: int
    miles: float = 0.0
    load: List[int] = None  # package IDs
    remaining: List[int] = None
    log: List[Tuple[int, str]] = None

    def __post_init__(self):
        if self.load is None:
            self.load = []
        if self.remaining is None:
            self.remaining = []
        if self.log is None:
            self.log = []


# =============================
# CSV loading
# =============================

def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().replace("#", "").split())


def load_packages_csv(path: str) -> HashTable:
    table = HashTable()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return table
    header = [h.strip() for h in rows[0]]

    def find_col(name_fragments: List[str]) -> int:
        # Return the first column index that contains all fragments
        for i, h in enumerate(header):
            hnorm = _normalize(h)
            ok = True
            for frag in name_fragments:
                if frag not in hnorm:
                    ok = False
                    break
            if ok:
                return i
        return -1

    c_id   = find_col(["package", "id"])
    c_addr = find_col(["address"])
    c_city = find_col(["city"])
    c_state= find_col(["state"])
    c_zip  = find_col(["zip"])
    c_dead = find_col(["delivery", "deadline"])
    c_wgh  = find_col(["weight"])  # handles "weight kilo"
    c_note = find_col(["special", "notes"]) if find_col(["special", "notes"]) != -1 else -1

    for r in rows[1:]:
        if len(r) <= max(c_id, c_addr, c_city, c_state, c_zip, c_dead, c_wgh):
            continue
        pid = int(r[c_id].strip())
        addr = r[c_addr].strip()
        city = r[c_city].strip()
        state= r[c_state].strip()
        zipc = r[c_zip].strip()
        dead = r[c_dead].strip()
        w    = int(r[c_wgh].strip()) if r[c_wgh].strip() else 0
        notes= r[c_note].strip() if c_note != -1 and c_note < len(r) else ""
        p = Package(
            id=pid,
            address=addr,
            city=city,
            state=state,
            zip=zipc,
            deadline_min=parse_clock(dead),
            weight=w,
            notes=notes,
        )
        table.insert(pid, p)
    return table


def load_distance_table_csv(path: str) -> Tuple[List[str], List[List[float]]]:
    rows: List[List[str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for r in reader:
            if any(cell.strip() for cell in r):
                rows.append(r)
    # find header with location labels
    header_idx = None
    for i, r in enumerate(rows):
        nonempty = [c for c in r if c.strip()]
        if len(nonempty) >= 3:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Distance table header not found")

    header = rows[header_idx]
    loc_labels = [c.strip() for c in header[1:] if c.strip()]
    n = len(loc_labels)

    mat: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        mat[i][i] = 0.0

    row_i = 0
    for r in rows[header_idx + 1:]:
        r = r + [""] * max(0, (n + 1) - len(r))
        row_label = r[0].strip()
        if not row_label:
            continue
        if row_i >= n:
            break
        for j in range(1, n + 1):
            cell = r[j].strip()
            if not cell:
                continue
            try:
                val = float(cell)
            except ValueError:
                continue
            i_idx = row_i
            j_idx = j - 1
            mat[i_idx][j_idx] = val
            mat[j_idx][i_idx] = val
        row_i += 1

    # Fill missing by symmetry
    for i in range(n):
        for j in range(n):
            if mat[i][j] is None:
                mat[i][j] = mat[j][i] if mat[j][i] is not None else (0.0 if i == j else 0.0)

    return loc_labels, mat  # square matrix


# =============================
# Address indexing
# =============================

def build_address_index(loc_labels: List[str]) -> Tuple[int, HashTable]:
    # Find hub and build normalized label → index map in a HashTable
    hub_idx = -1
    addr_ht = HashTable(max(53, len(loc_labels) * 2))
    for i, lab in enumerate(loc_labels):
        norm = _normalize(lab)
        addr_ht.insert(norm, i)
        if "western governors university" in norm:
            hub_idx = i
    if hub_idx == -1:
        raise ValueError("Hub not found in distance labels")
    return hub_idx, addr_ht


def map_package_addresses(packages: HashTable, loc_labels: List[str], addr_ht: HashTable) -> None:
    # Attach matrix index to each package by normalized lookup or fallback scan
    for _, p in packages.items():
        norm = _normalize(p.address)
        idx = addr_ht.get(norm)
        if idx is None:
            # fallback: try substring numeric match
            tokens = p.address.strip().split()
            num = tokens[0] if tokens else ""
            found = None
            for i, lab in enumerate(loc_labels):
                if f" {num} " in f" {lab} ":
                    found = i
                    break
            if found is None:
                raise ValueError(f"Could not map address for package {p.id}: {p.address}")
            p.index = found
        else:
            p.index = int(idx)


# =============================
# Constraints
# =============================

GROUP_G1 = [13, 14, 15, 16, 19, 20]
TRUCK2_ONLY = [3, 18, 36, 38]
DELAY_905 = [6, 25, 28, 32]


def apply_constraints(packages: HashTable) -> None:
    for _, p in packages.items():
        # defaults
        p.status = "at hub"
        p.earliest_ready = START_MIN
        p.truck_only = None
        p.group_id = None
    for pid in DELAY_905:
        p = packages.get(pid)
        if p is not None:
            p.earliest_ready = max(p.earliest_ready, parse_clock("9:05 AM"))
    for pid in TRUCK2_ONLY:
        p = packages.get(pid)
        if p is not None:
            p.truck_only = 2
    for pid in GROUP_G1:
        p = packages.get(pid)
        if p is not None:
            p.group_id = "G1"
    p9 = packages.get(9)
    if p9 is not None:
        p9.earliest_ready = max(p9.earliest_ready, TEN_TWENTY)


# =============================
# Router
# =============================

class Router:
    def __init__(self, labels: List[str], dist: List[List[float]], hub_idx: int, packages: HashTable):
        self.labels = labels
        self.D = dist
        self.hub = hub_idx
        self.pkgs = packages

    def travel_minutes(self, i: int, j: int) -> int:
        miles = self.D[i][j]
        return int(round(miles / SPEED_MPH * 60.0))

    def _deliver_here(self, truck: Truck, idx: int):
        delivered_ids: List[int] = []
        # deliver all matching address currently on board
        still: List[int] = []
        for pid in truck.remaining:
            p = self.pkgs.get(pid)
            if p is None:
                continue
            if p.index == idx and p.earliest_ready <= truck.depart_min:
                p.status = "delivered"
                p.time_delivered = truck.depart_min
                p.truck_delivered = truck.id
                delivered_ids.append(pid)
            else:
                still.append(pid)
        truck.remaining = still
        if delivered_ids:
            truck.log.append((truck.depart_min, f"Delivered {sorted(delivered_ids)} at {self.labels[idx]}"))

    def _ready_ids(self, truck: Truck) -> List[int]:
        ready: List[int] = []
        for pid in truck.remaining:
            p = self.pkgs.get(pid)
            if p is not None and p.earliest_ready <= truck.depart_min:
                ready.append(pid)
        return ready

    def _feasible_ids(self, truck: Truck, ready_ids: List[int]) -> List[int]:
        feas: List[int] = []
        for pid in ready_ids:
            p = self.pkgs.get(pid)
            if p is None:
                continue
            arrive = truck.depart_min + self.travel_minutes(truck.location_idx, int(p.index))  # type: ignore
            if p.deadline_min == EOD_MIN or arrive <= p.deadline_min:
                feas.append(pid)
        return feas if feas else ready_ids

    def _choose_next(self, truck: Truck, cand: List[int]) -> Optional[int]:
        if not cand:
            return None
        best_pid = cand[0]
        best_key = (float("inf"), 3, 10**9)
        for pid in cand:
            p = self.pkgs.get(pid)
            if p is None:
                continue
            dist = self.D[truck.location_idx][int(p.index)]  # type: ignore
            key = (dist, p.deadline_rank(), pid)
            if key < best_key:
                best_key = key
                best_pid = pid
        return best_pid

    def route_truck(self, truck: Truck):
        # mark en route
        for pid in truck.load:
            p = self.pkgs.get(pid)
            if p is None:
                continue
            p.status = "en route"
            p.truck_loaded = truck.id
            if p.time_loaded is None:
                p.time_loaded = truck.depart_min
        truck.remaining = list(truck.load)
        truck.log.append((truck.depart_min, f"Depart hub with {sorted(truck.load)}"))

        while len(truck.remaining) > 0:
            ready = self._ready_ids(truck)
            if not ready:
                # wait until the next earliest ready among remaining
                next_ready = min(self.pkgs.get(pid).earliest_ready for pid in truck.remaining if self.pkgs.get(pid) is not None)
                truck.log.append((truck.depart_min, f"Waiting until {next_ready} for readiness"))
                truck.depart_min = max(truck.depart_min, next_ready)
                continue
            feas = self._feasible_ids(truck, ready)
            nxt = self._choose_next(truck, feas)
            if nxt is None:
                break
            p = self.pkgs.get(nxt)
            if p is None:
                # remove and continue
                truck.remaining = [x for x in truck.remaining if x != nxt]
                continue
            tm = self.travel_minutes(truck.location_idx, int(p.index))  # type: ignore
            truck.miles += self.D[truck.location_idx][int(p.index)]  # type: ignore
            truck.depart_min += tm
            truck.location_idx = int(p.index)  # type: ignore
            self._deliver_here(truck, truck.location_idx)

        # return to hub
        if truck.location_idx != self.hub:
            tm = self.travel_minutes(truck.location_idx, self.hub)
            truck.miles += self.D[truck.location_idx][self.hub]
            truck.depart_min += tm
            truck.location_idx = self.hub
            truck.log.append((truck.depart_min, "Returned to hub"))

    # -------------- Loading --------------
    def _group_onboard(self, chosen: List[int]) -> bool:
        # Check if all G1 members are present when any are chosen
        any_member = any(pid in chosen for pid in GROUP_G1)
        if not any_member:
            return True
        for pid in GROUP_G1:
            if pid not in chosen:
                return False
        return True

    def greedy_load(self, truck_id: int, now_min: int, already_assigned: List[int]) -> List[int]:
        cand: List[Package] = []
        # collect ready and allowed candidates
        for _, p in self.pkgs.items():
            if p.id in already_assigned:
                continue
            if p.earliest_ready > now_min:
                continue
            if p.truck_only is not None and p.truck_only != truck_id:
                continue
            cand.append(p)
        # sort by deadline rank then hub distance
        def _hub_dist(pkg: Package) -> float:
            return self.D[self.hub][int(pkg.index)]  # type: ignore
        cand.sort(key=lambda x: (x.deadline_rank(), _hub_dist(x)))

        chosen: List[int] = []

        # If any G1 member is available and capacity allows, load the whole group
        if any(p.id in GROUP_G1 for p in cand):
            if len(GROUP_G1) <= CAPACITY:
                # ensure all are available and not assigned yet
                can_take = True
                for gid in GROUP_G1:
                    gp = self.pkgs.get(gid)
                    if gp is None or gp.earliest_ready > now_min or gid in already_assigned:
                        can_take = False
                        break
                if can_take:
                    for gid in GROUP_G1:
                        chosen.append(gid)
                        already_assigned.append(gid)

        for p in cand:
            if len(chosen) >= CAPACITY:
                break
            if p.id in chosen:
                continue
            if p.group_id == "G1":
                continue  # already handled as a block
            chosen.append(p.id)
            already_assigned.append(p.id)

        return chosen


# =============================
# Supervisor helpers
# =============================

def fmt_time(mins: Optional[int]) -> str:
    if mins is None:
        return "--:--"
    t = BASE_DAY + timedelta(minutes=mins)
    return t.strftime("%I:%M %p").lstrip("0")


def status_at_time(p: Package, tmin: int) -> str:
    if p.time_loaded is None or tmin < p.time_loaded:
        return "at the hub"
    if p.time_delivered is not None and tmin >= p.time_delivered:
        return f"delivered at {fmt_time(p.time_delivered)}"
    return "en route"


def print_snapshot(packages: HashTable, tmin: int) -> None:
    print(f"=== Status snapshot at {fmt_time(tmin)} ===")
    for truck_id in (1, 2, 3):
        rows: List[Tuple[int, str]] = []
        for _, p in packages.items():
            if p.truck_loaded == truck_id and p.time_loaded is not None and p.time_loaded <= tmin:
                rows.append((p.id, status_at_time(p, tmin)))
        rows.sort(key=lambda x: x[0])
        print(f"Truck {truck_id}:")
        if not rows:
            print("  (no packages loaded by this time)")
        else:
            for pid, st in rows:
                print(f"  Pkg {pid:>2}: {st}")

def parse_user_time(s: str) -> Optional[int]:
    """Accepts 'HH:MM' (24h) or 'H:MM AM/PM'. Returns minutes-from-8:00 or None."""
    if not s:
        return None
    s = s.strip()
    try:
        # allow AM/PM
        if s.lower().endswith(("am", "pm")):
            return parse_clock(s)
        # allow 24h HH:MM
        hh, mm = s.split(":")
        h = int(hh); m = int(mm)
        return _to_minutes(BASE_DAY.replace(hour=h, minute=m))
    except Exception:
        return None

def deadline_to_str(p: Package) -> str:
    return "EOD" if p.deadline_min == EOD_MIN else fmt_time(p.deadline_min)

def print_all_status(packages: HashTable, tmin: int) -> None:
    """Show ALL packages with address, deadline, truck, and status at time tmin."""
    print(f"\n=== All packages at {fmt_time(tmin)} ===")
    print(f"{'ID':>3}  {'Address':35}  {'Deadline':8}  {'Truck':5}  Status")

    ids = [pid for pid, _ in packages.items()]
    ids.sort()

    for pid in ids:
        p = packages.get(pid)
        if p is None:
            continue
        status = status_at_time(p, tmin)  # 'at the hub' | 'en route' | 'delivered at HH:MM'
        # choose delivered truck if delivered by t, else the loaded truck
        truck = p.truck_delivered if (p.time_delivered is not None and tmin >= p.time_delivered) else p.truck_loaded
        truck_str = str(truck) if truck is not None else "-"
        addr = f"{p.address}, {p.city} {p.zip}"
        if len(addr) > 35:
            addr = addr[:32] + "..."
        print(f"{pid:>3}  {addr:35}  {deadline_to_str(p):8}  {truck_str:5}  {status}")

def print_one_status(packages: HashTable, pid: int, tmin: int) -> None:
    p = packages.get(pid)
    if p is None:
        print("No such package.")
        return
    status = status_at_time(p, tmin)
    truck = p.truck_delivered if (p.time_delivered is not None and tmin >= p.time_delivered) else p.truck_loaded
    truck_str = str(truck) if truck is not None else "-"
    print(f"\nPackage {pid}: {status}")
    print(f"Address:  {p.address}, {p.city} {p.zip}")
    print(f"Deadline: {deadline_to_str(p)}")
    print(f"Truck:    {truck_str}")
    if p.time_delivered is not None and tmin >= p.time_delivered:
        print(f"Delivered at: {fmt_time(p.time_delivered)}")


# =============================
# Simulation entry point
# =============================

def run_simulation(pkg_csv: str, dist_csv: str) -> None:
    packages = load_packages_csv(pkg_csv);
    apply_constraints(packages);

    labels, D = load_distance_table_csv(dist_csv)
    hub_idx, addr_ht = build_address_index(labels)
    map_package_addresses(packages, labels, addr_ht)

    router = Router(labels, D, hub_idx, packages)

    assigned: List[int] = []

    # Truck 1 and 2 leave at 8:00
    T1 = Truck(id=1, depart_min=START_MIN, location_idx=hub_idx)
    T1.load = router.greedy_load(1, T1.depart_min, assigned)

    T2 = Truck(id=2, depart_min=START_MIN, location_idx=hub_idx)
    T2.load = router.greedy_load(2, T2.depart_min, assigned)

    router.route_truck(T1)
    router.route_truck(T2)

    # Reset assigned to only delivered, so undelivered can be reloaded
    assigned = [p.id for _, p in packages.items() if p.status == "delivered"]

    # Truck 3 leaves when first driver returns
    first_free = T1.depart_min if T1.depart_min <= T2.depart_min else T2.depart_min
    T3 = Truck(id=3, depart_min=first_free, location_idx=hub_idx)
    T3.load = router.greedy_load(3, T3.depart_min, assigned)
    router.route_truck(T3)

    # Reset assigned after T3
    assigned = [p.id for _, p in packages.items() if p.status == "delivered"]

    # If anything remains, relaunch trucks as they free up
    while True:
        # Stop if everything is delivered
        remaining_ids: List[int] = []
        for _, p in packages.items():
            if p.status != "delivered":
                remaining_ids.append(p.id)
        if not remaining_ids:
            break

        # Try each truck in order of availability; only route if it actually has a load
        trio = [(T1.depart_min, T1), (T2.depart_min, T2), (T3.depart_min, T3)]
        trio.sort(key=lambda x: x[0])
        progressed = False

        for _, tr in trio:
            # Exclude already delivered packages from future loads
            assigned = [p.id for _, p in packages.items() if p.status == "delivered"]
            tr.location_idx = hub_idx  # ensure we are loading at the hub
            load = router.greedy_load(tr.id, tr.depart_min, assigned)
            if load:
                tr.load = load
                router.route_truck(tr)
                progressed = True
                # After routing one truck, restart the outer loop to re-evaluate state
                break

        if progressed:
            continue

        # No truck could take a load now. Advance time safely to avoid spinning.
        current_min = min(T1.depart_min, T2.depart_min, T3.depart_min)
        next_ready = None
        for _, p in packages.items():
            if p.status != "delivered" and p.earliest_ready > current_min:
                next_ready = p.earliest_ready if next_ready is None else min(next_ready, p.earliest_ready)
        next_truck = None
        for tmin, _ in trio:
            if tmin > current_min:
                next_truck = tmin if next_truck is None else min(next_truck, tmin)

        # Choose the earliest meaningful time to move forward
        if next_ready is None and next_truck is None:
            # Last resort: bump the earliest truck by 1 minute
            if T1.depart_min <= T2.depart_min and T1.depart_min <= T3.depart_min:
                T1.depart_min += 1
            elif T2.depart_min <= T1.depart_min and T2.depart_min <= T3.depart_min:
                T2.depart_min += 1
            else:
                T3.depart_min += 1
        else:
            advance = next_ready if next_truck is None else (next_truck if next_ready is None else min(next_ready, next_truck))
            # Set the earliest truck to that time
            if T1.depart_min <= T2.depart_min and T1.depart_min <= T3.depart_min:
                T1.depart_min = max(T1.depart_min, advance)
            elif T2.depart_min <= T1.depart_min and T2.depart_min <= T3.depart_min:
                T2.depart_min = max(T2.depart_min, advance)
            else:
                T3.depart_min = max(T3.depart_min, advance)

    total_miles = T1.miles + T2.miles + T3.miles

    print("=== WGUPS Routing Summary ===")
    for T in [T1, T2, T3]:
        print(f"Truck {T.id}: miles={T.miles:.1f}, finish={fmt_time(T.depart_min)}")
    print(f"Total miles: {total_miles:.1f}")

    print("Sample deliveries (first 10 by ID):")
    # build a list of IDs
    ids: List[int] = [pid for pid, _ in packages.items()]
    ids.sort()
    for pid in ids[:10]:
        p = packages.get(pid)
        print(f"Pkg {pid}: {p.status}, delivered={fmt_time(p.time_delivered)}")

    # ===== Text UI (Requirement D) =====
    while True:
        print("\n== WGUPS Menu ==")
        print("1) View ALL packages at a time")
        print("2) Look up a package at a time")
        print("3) Exit")
        choice = input("Choose 1-3: ").strip()

        if choice == "1":
            t = parse_user_time(input("Enter time (e.g., 09:45 or 9:45 AM): "))
            if t is None:
                print("Invalid time. Try again.")
                continue
            print_all_status(packages, t)
            print(f"\nTotal miles (all trucks): {total_miles:.1f}")

        elif choice == "2":
            pid_s = input("Package ID (1-40): ").strip()
            if not pid_s.isdigit():
                print("Enter a numeric package ID.")
                continue
            t = parse_user_time(input("Enter time (e.g., 09:45 or 9:45 AM): "))
            if t is None:
                print("Invalid time. Try again.")
                continue
            print_one_status(packages, int(pid_s), t)
            print(f"\nTotal miles (all trucks): {total_miles:.1f}")

        elif choice == "3":
            break
        else:
            print("Enter 1, 2, or 3.")


if __name__ == "__main__":
    run_simulation(
        pkg_csv="WGUPS_Package_File.csv",
        dist_csv="WGUPS_Distance_Table.csv",
    )
