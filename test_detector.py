from ai.detector import get_or_create_detector
import numpy as np

print("getting")
det = get_or_create_detector(session_id=1, capture_dir='captures')
print("got")
print("analyzing")
res = det.analyze_frame('data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=')
print("done")
print(res.has_violation)
