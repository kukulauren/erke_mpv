import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# === CONFIGURATION ===
MODEL_PATH = "best.pt"
VIDEO_PATH = "Untitled.mp4"
OUTPUT_PATH = "output_result.mp4"

# Distance thresholds
SCANNER_PHONE_DISTANCE = 80      # Phone near scanner = payment
SCANNER_ITEM_DISTANCE = 100      # REDUCED! Scanner must be very close to item

# Scanner movement
SCANNER_MOVEMENT_THRESHOLD = 3   # Pixels - scanner moving if > this

# Timing
PAYMENT_COMPLETE_TIME = 1.0
SCAN_COOLDOWN = 1.5              # Seconds before same item can be scanned again

# Customer
OVERLAP_THRESHOLD = 0.3
CUSTOMER_DWELL_TIME = 3.0

CLASS_NAMES = {0: 'cashier', 1: 'customer', 2: 'scanner', 3: 'item', 4: 'phone', 5: 'cash', 6: 'counter'}
CLASS_COLORS = {
    'cashier': (0, 255, 0),
    'customer': (255, 0, 0),
    'scanner': (0, 0, 255),
    'item': (255, 255, 0),
    'phone': (255, 0, 255),
    'cash': (0, 255, 255),
    'counter': (128, 128, 128)
}


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def boxes_overlap(box1, box2):
    """Check if two boxes overlap at all"""
    if box1[0] > box2[2] or box2[0] > box1[2]:  # No horizontal overlap
        return False
    if box1[1] > box2[3] or box2[1] > box1[3]:  # No vertical overlap
        return False
    return True


def get_box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


class RetailAnalytics:
    def __init__(self, fps=30):
        self.fps = fps

        # Scanner tracking
        self.scanner_positions = {}  # scanner_id: [last_positions]
        self.scanner_moving = {}     # scanner_id: bool

        # Item scanning
        self.scanned_items = []
        self.last_scan_time = {}     # item_id: last_scan_time
        self.current_overlaps = []

        # Payment
        self.payment_in_progress = None
        self.completed_payments = []
        self.phone_tracks = defaultdict(list)
        self.payment_times = []

        # Customer
        self.customers_at_counter = {}
        self.customer_visits = []
        self.service_times = []

    def update_scanner_movement(self, scanners, current_time):
        """Track if scanner is moving"""
        scanner_status = {}

        for scanner in scanners:
            scanner_id = scanner.get('track_id') or 0
            current_center = scanner['center']

            if scanner_id not in self.scanner_positions:
                self.scanner_positions[scanner_id] = [current_center]
                scanner_status[scanner_id] = {'moving': False, 'speed': 0}
            else:
                prev_center = self.scanner_positions[scanner_id][-1]
                movement = get_distance(current_center, prev_center)

                is_moving = movement > SCANNER_MOVEMENT_THRESHOLD
                self.scanner_moving[scanner_id] = is_moving

                scanner_status[scanner_id] = {
                    'moving': is_moving,
                    'speed': movement,
                    'center': current_center
                }

                # Keep last 5 positions
                self.scanner_positions[scanner_id].append(current_center)
                if len(self.scanner_positions[scanner_id]) > 5:
                    self.scanner_positions[scanner_id].pop(0)

        return scanner_status

    def update_item_scanning(self, items, scanners, scanner_status, current_time):
        """
        NEW LOGIC: Scanner MOVES and overlaps/near Item = Item Scanned

        Conditions for scan:
        1. Scanner bbox overlaps Item bbox OR distance < threshold
        2. Scanner was MOVING (approaching the item)
        3. Cooldown passed for this item
        """
        events = []
        self.current_overlaps = []

        for scanner in scanners:
            scanner_id = scanner.get('track_id') or 0
            status = scanner_status.get(scanner_id, {})
            is_moving = status.get('moving', False)

            scanner_box = scanner['box']
            scanner_center = scanner['center']

            for item in items:
                item_id = item.get('track_id') or id(item)
                item_box = item['box']
                item_center = item['center']

                # Check overlap (bbox intersection)
                has_overlap = boxes_overlap(scanner_box, item_box)
                iou = get_box_iou(scanner_box, item_box) if has_overlap else 0

                # Also check distance
                dist = get_distance(scanner_center, item_center)
                is_close = dist < SCANNER_ITEM_DISTANCE

                # Either overlap OR very close
                is_scanning_position = has_overlap or is_close

                self.current_overlaps.append({
                    'item_id': item_id,
                    'scanner_id': scanner_id,
                    'distance': dist,
                    'has_overlap': has_overlap,
                    'iou': iou,
                    'is_close': is_close,
                    'scanner_moving': is_moving
                })

                # SCAN DETECTION: Scanner moved and is now overlapping/close to item
                if is_scanning_position and is_moving:
                    # Check cooldown
                    last_scan = self.last_scan_time.get(item_id, 0)
                    if current_time - last_scan > SCAN_COOLDOWN:
                        # SCANNED!
                        self.scanned_items.append({
                            'time': current_time,
                            'item_id': item_id,
                            'scanner_id': scanner_id,
                            'distance': dist,
                            'iou': iou
                        })
                        self.last_scan_time[item_id] = current_time
                        events.append(f"✓ ITEM #{item_id} SCANNED!")

        return events

    def update_payment_scanning(self, phones, scanners, current_time):
        """Phone near scanner for 1s = payment complete (working well)"""
        events = []
        phone_near_scanner = False

        for phone in phones:
            phone_id = phone.get('track_id') or id(phone)

            self.phone_tracks[phone_id].append({
                'position': phone['center'],
                'time': current_time
            })
            if len(self.phone_tracks[phone_id]) > 60:
                self.phone_tracks[phone_id].pop(0)

            for scanner in scanners:
                dist = get_distance(phone['center'], scanner['center'])

                if dist < SCANNER_PHONE_DISTANCE:
                    phone_near_scanner = True

                    if self.payment_in_progress is None:
                        self.payment_in_progress = {
                            'start_time': current_time,
                            'phone_id': phone_id
                        }
                        events.append("📱 PAYMENT STARTED...")
                    else:
                        duration = current_time - self.payment_in_progress['start_time']
                        if duration >= PAYMENT_COMPLETE_TIME:
                            self.completed_payments.append({
                                'time': current_time,
                                'phone_id': phone_id,
                                'duration': duration
                            })
                            self.payment_times.append(duration)
                            events.append("✓ PAYMENT COMPLETE (Mobile)")
                            self.payment_in_progress = None
                    break

        if not phone_near_scanner and self.payment_in_progress:
            self.payment_in_progress = None

        return events

    def update_customer_at_counter(self, customers, counters, current_time):
        """Customer bbox overlapping counter"""
        events = []
        active_ids = set()

        for customer in customers:
            customer_id = customer.get('track_id') or id(customer)
            active_ids.add(customer_id)

            for counter in counters:
                has_overlap = boxes_overlap(customer['box'], counter['box'])

                if has_overlap:
                    if customer_id not in self.customers_at_counter:
                        self.customers_at_counter[customer_id] = {
                            'arrival_time': current_time,
                            'counted': False
                        }
                        events.append(f"👤 CUSTOMER #{customer_id} AT COUNTER")
                    else:
                        dwell = current_time - self.customers_at_counter[customer_id]['arrival_time']
                        if dwell >= CUSTOMER_DWELL_TIME and not self.customers_at_counter[customer_id]['counted']:
                            self.customers_at_counter[customer_id]['counted'] = True
                            self.customer_visits.append({'customer_id': customer_id})
                    break

        for cid in list(self.customers_at_counter.keys()):
            if cid not in active_ids:
                dwell = current_time - self.customers_at_counter[cid]['arrival_time']
                self.service_times.append(dwell)
                events.append(f"👤 CUSTOMER #{cid} LEFT ({dwell:.1f}s)")
                del self.customers_at_counter[cid]

        return events

    def get_display_stats(self):
        return [
            f"Items Scanned: {len(self.scanned_items)}",
            f"Payments: {len(self.completed_payments)}",
            f"At Counter: {len(self.customers_at_counter)}"
        ]


def process_video(model_path, video_path, output_path, conf_threshold):
    print("Loading model...")
    model = YOLO(model_path)

    print("Opening video...")
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open. Check codec & headless environment.")
        
    analytics = RetailAnalytics(fps=fps)

    frame_count = 0
    start_time = time.time()

    print(f"Processing {total_frames} frames at {fps} FPS...")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # Run detection
        results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)

        detections = {k: [] for k in CLASS_NAMES.values()}

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else None

                detections[cls_name].append({
                    'box': xyxy,
                    'conf': conf,
                    'track_id': track_id,
                    'center': get_center(xyxy)
                })

        # DEBUG: Print every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\nProgress: {progress:.1f}% ({frame_count}/{total_frames})")
            print(f"  Scanners: {len(detections['scanner'])}, Items: {len(detections['item'])}")

            for scanner in detections['scanner']:
                sid = scanner.get('track_id') or 0
                moving = analytics.scanner_moving.get(sid, False)
                print(f"  Scanner #{sid}: moving={moving}")
                for item in detections['item']:
                    dist = get_distance(scanner['center'], item['center'])
                    overlap = boxes_overlap(scanner['box'], item['box'])
                    iid = item.get('track_id') or 'no_id'
                    print(f"    -> Item #{iid}: dist={dist:.0f}px, overlap={overlap}")

        # === ANALYTICS ===
        all_events = []

        # 1. Track scanner movement
        scanner_status = analytics.update_scanner_movement(detections['scanner'], current_time)

        # 2. Item scanning (scanner moves + overlaps item)
        scan_events = analytics.update_item_scanning(
            detections['item'],
            detections['scanner'],
            scanner_status,
            current_time
        )
        all_events.extend(scan_events)

        # 3. Payment
        payment_events = analytics.update_payment_scanning(
            detections['phone'],
            detections['scanner'],
            current_time
        )
        all_events.extend(payment_events)

        # 4. Customer at counter
        customer_events = analytics.update_customer_at_counter(
            detections['customer'],
            detections['counter'],
            current_time
        )
        all_events.extend(customer_events)

        # === DRAW ===
        for cls_name, dets in detections.items():
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            for det in dets:
                box = det['box'].astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

                label = cls_name
                if det['track_id']:
                    label = f"{cls_name} #{det['track_id']}"
                cv2.putText(frame, label, (box[0], box[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw scanner zone (smaller circle)
        for scanner in detections['scanner']:
            center = tuple(map(int, scanner['center']))
            scanner_id = scanner.get('track_id') or 0
            is_moving = analytics.scanner_moving.get(scanner_id, False)

            # Color based on movement
            zone_color = (0, 255, 0) if is_moving else (0, 255, 255)  # Green if moving, Yellow if still
            cv2.circle(frame, center, SCANNER_ITEM_DISTANCE, zone_color, 2)

            status = "MOVING" if is_moving else "STILL"
            cv2.putText(frame, f"SCAN ZONE ({status})",
                       (center[0] - 60, center[1] - SCANNER_ITEM_DISTANCE - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

        # Draw overlap info
        y = 30
        cv2.putText(frame, f"Scanner Zone: {SCANNER_ITEM_DISTANCE}px (GREEN=moving)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y += 25

        for overlap in analytics.current_overlaps:
            item_id = overlap['item_id']
            dist = overlap['distance']
            has_overlap = overlap['has_overlap']
            is_moving = overlap['scanner_moving']

            if has_overlap:
                color = (0, 255, 0)  # Green - overlapping
                status = "OVERLAP!"
            elif overlap['is_close']:
                color = (0, 255, 255)  # Yellow - close
                status = "CLOSE"
            else:
                color = (128, 128, 128)
                status = ""

            scanner_status = "MOVING" if is_moving else "still"
            cv2.putText(frame, f"Item #{item_id}: {dist:.0f}px {status} (scanner {scanner_status})",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y += 20

        # Phone tracks
        for phone_id, track_data in analytics.phone_tracks.items():
            if len(track_data) > 1:
                pts = np.array([t['position'] for t in track_data], dtype=np.int32)
                cv2.polylines(frame, [pts], False, CLASS_COLORS['phone'], 2)

        # Payment progress
        if analytics.payment_in_progress:
            progress = (current_time - analytics.payment_in_progress['start_time']) / PAYMENT_COMPLETE_TIME
            progress = min(progress, 1.0)
            cv2.rectangle(frame, (10, y + 10), (10 + int(200 * progress), y + 30), (255, 0, 255), -1)
            cv2.rectangle(frame, (10, y + 10), (210, y + 30), (255, 255, 255), 2)
            cv2.putText(frame, f"Payment: {progress*100:.0f}%", (10, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Events (right side)
        event_y = 30
        for event in all_events:
            cv2.putText(frame, event, (width - 400, event_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            event_y += 25
            print(f"[{current_time:.1f}s] {event}")

        # Stats bar
        stats = analytics.get_display_stats()
        cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
        x = 20
        for stat in stats:
            cv2.putText(frame, stat, (x, height - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            x += 220

        out.write(frame)

    cap.release()
    out.release()
    time.sleep(1)

    processing_time = time.time() - start_time

    # Report
    print("\n" + "="*60)
    print("📊 RETAIL ANALYTICS REPORT")
    print("="*60)
    print(f"\n📋 SUMMARY:")
    print(f"  • Total Items Scanned: {len(analytics.scanned_items)}")
    print(f"  • Total Payments: {len(analytics.completed_payments)}")
    print(f"  • Customers Served: {len(analytics.customer_visits)}")
    print(f"\n📁 Output: {output_path}")
    print(f"⏱️ Time: {processing_time:.1f}s")
    print("="*60)


if __name__ == "__main__":
    process_video(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH)