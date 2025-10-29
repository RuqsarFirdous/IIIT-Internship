import cv2

# Load input and output videos
input_video = cv2.VideoCapture('input_video.mp4')
output_video = cv2.VideoCapture('output_video.mp4')

# Get video properties from input video
fps = int(input_video.get(cv2.CAP_PROP_FPS))
width = 640
height = 360

# Define output video writer (side-by-side)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('comparison_video.mp4', fourcc, fps, (width * 2, height))

while True:
    ret1, frame1 = input_video.read()
    ret2, frame2 = output_video.read()

    if not ret1 or not ret2:
        break  # stop when any video ends

    # Resize both frames
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # Combine side-by-side
    combined = cv2.hconcat([frame1, frame2])

    # Write frame to output
    out.write(combined)

    # Optional: display live preview
    cv2.imshow('Input (Left) | Output (Right)', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
input_video.release()
output_video.release()
out.release()
cv2.destroyAllWindows()

print("✅ comparison_video.mp4 has been saved successfully!")

