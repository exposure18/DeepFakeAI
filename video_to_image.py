import cv2
import os


def extract_frames(video_path, output_folder, frame_rate=10):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str): The path to the video file.
        output_folder (str): The folder to save the extracted images.
        frame_rate (int): Saves every nth frame.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_rate == 0:
            frame_filename = f"{os.path.basename(video_path).split('.')[0]}_{saved_frame_count}.jpg"
            output_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {os.path.basename(video_path)}.")


def extract_frames_from_folder(source_folder, output_folder, frame_rate=10):
    """
    Iterates through all video files in a folder, extracts frames, and saves them.

    Args:
        source_folder (str): The folder containing the video files.
        output_folder (str): The folder to save the extracted images.
        frame_rate (int): Saves every nth frame.
    """
    video_files = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]

    print(f"Starting to process videos from {os.path.basename(source_folder)}...")
    for video_name in video_files:
        video_path = os.path.join(source_folder, video_name)
        extract_frames(video_path, output_folder, frame_rate)

    print(f"Finished processing all videos in {os.path.basename(source_folder)}.")


if __name__ == "__main__":
    # --- Example Usage ---
    # Replace the base path with the actual path on your computer
    base_path = r"C:\Users\Ricky Rodrigues\Downloads\archive (1)\FaceForensics++_C23"

    real_videos_folder = os.path.join(base_path, "original")
    fake_videos_folders = [
        os.path.join(base_path, "Deepfakes"),
        os.path.join(base_path, "Face2Face"),
        os.path.join(base_path, "FaceShifter"),
        os.path.join(base_path, "FaceSwap"),
        os.path.join(base_path, "NeuralTextures")
    ]

    # Make sure you have created the 'data/real' and 'data/fake' folders in your project
    real_images_folder = "data/real"
    fake_images_folder = "data/fake"

    # Process all real videos
    extract_frames_from_folder(real_videos_folder, real_images_folder)

    # Process all fake videos
    for folder in fake_videos_folders:
        extract_frames_from_folder(folder, fake_images_folder)

    print("\nAll videos have been processed! Your images are ready in the 'data' folder.")