# WGUPS Routing Program – C950 Task 2 (Python)

Efficient package delivery for the Western Governors University Parcel Service (WGUPS).  
This program simulates three trucks, two drivers, and forty packages. It respects deadlines and special constraints and delivers all packages while keeping the combined mileage under 140 miles.

## Highlights
- Custom **HashTable** with separate chaining (no Python dicts in core routing and storage)
- Greedy **nearest neighbor** routing with deadline tie breaks
- Text **UI** to query package status at any time
- Constraints implemented:
  - 9:00 AM and 10:30 AM deadlines
  - Truck 2 only packages
  - Grouped packages {13, 14, 15, 16, 19, 20} travel together
  - Flight delays until 9:05 AM
  - Package 9 address becomes valid at 10:20 AM

---

## Quick start

### Requirements
- Python 3.10 or newer
- The following files in the project folder:
  - `main.py`
  - `WGUPS_Package_File.csv`
  - `WGUPS_Distance_Table.csv`

### Run
```bash
python main.py



CSV path issues
Keep the CSV files in the same folder as main.py or pass absolute paths.

GitHub push rejected as non fast forward
Run git pull --rebase origin main, resolve conflicts, then git push.
