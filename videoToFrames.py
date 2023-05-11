import os
import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QLineEdit,
    QLabel,
    QCheckBox,
    QStyle,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QByteArray, QIODevice
from PyQt6.QtMultimedia import QMediaPlayer, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
import numpy as np

# import QBuffer
from PyQt6.QtCore import QBuffer
import ffmpeg
from PIL import Image

# import from .env file
from dotenv import load_dotenv

load_dotenv()
import os
import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO
import matplotlib as plt
import requests
import imageio
from PyQt6.QtCore import QUrl

auth_token = os.getenv("AUTH_TOKEN")
prompt = ""
negative_prompt = ""
device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.bfloat16, use_auth_token=auth_token
)
pipe.to(device)

preview_mp4 = None


def is_video_correct(video_file_path):
    try:
        (ffmpeg.input(video_file_path).output("null", f="null").run())
    except ffmpeg._run.Error:
        print("corrupt video")
        return False
    else:
        print("video is fine")
        return True


# TODO: 1. select custom face image and upload your face movement video from this code: https://colab.research.google.com/drive/1SgNfi6i0rDCX4TPNmpfQTzdFp3qQqSsj#scrollTo=czsWABcK_2KE
# TODO: 2. copy 3 frames of generated video for each face micics type (recognize with ai)
# TODO: 3. generate diffused frames from these mimics frames + outpainting in stablediffusion infinity and  into ZWX folder
# TODO: 4. run DreamBoth Stable diffusion on these from ZWX folder as ZWX object - https://colab.research.google.com/drive/1pNJSuDyBWTnF4-FGTUSDHFG5OtiuztGd -> Save model to use later
# TODO: 5. generate diffused frames from video using promot like 'iamge of ZWX person doing ... and negative prompts.. using saved model from step 4
# TODO: 6. generate video from these frames options from video from automatic111


# ! https://www.youtube.com/watch?v=XjObqq6we4U&t=474s
# ! https://www.youtube.com/watch?v=3wQBsFftbv8&list=PLcJsxVbg4Lo9EkoOjJtLDSHG9l0Ly__8j&index=6
class VideoProcessorThread(QThread):
    progress_signal = pyqtSignal(int)
    preview_signal = pyqtSignal(int)

    def __init__(self, video_file_path):
        super().__init__()
        self.video_file_path = video_file_path
        self.confirmed = False
        self.generated_preview_images = []
        self.prompt = ""
        self.negative_prompt = ""

    def set_prompts(self, prompt, negative_prompt):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        print("Prompts: ")

    def generate_diffused_frame(self, frame):
        print("Generating diffused frame.. prompt: " + str(prompt))
        with autocast(device):
            init_image = Image.open(BytesIO(frame)).convert("RGB")

            # 16:9
            init_image = init_image.resize((432, 768))
            # 2 times smaller
            # init_image = init_image.resize((216, 384))
            # 4 times smaller
            # init_image = init_image.resize((192, 128))

            # init_image = Image.open(BytesIO(frame)).convert("RGB")
            # init_image = init_image.resize((768, 512))

            generator = torch.Generator("cuda").manual_seed(1024)

            images = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.25,
                guidance_scale=5.5,
                generator=generator,
            ).images
            print("Diffused frame generated!")
            # save image to file
            images[0].save("fantasy_landscape.png")

            print("Diffused frame saved to file!")
            # show image
            # convert to numpy array

            self.generated_preview_images.append(images[0])

    def run(self):
        cap = cv2.VideoCapture(self.video_file_path)
        torch.cuda.empty_cache()

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a folder to store the frames
        video_name = os.path.splitext(os.path.basename(self.video_file_path))[0]

        frames_dir = f"{video_name}_frames"
        os.makedirs(frames_dir, exist_ok=True)

        test_frames = []

        frame_count = 0
        video_frames_count = 0
        loop_count = 0
        is_preview = False

        generation_count = 0

        while (
            cap.isOpened()
        ):  # Extract the frames from the video and save them as images
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_filename = f"{video_name}_frame{frame_count}.png"
            frame_filepath = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame)

            # convert fram to bytes

            if height % 32 != 0:
                frame = cv2.resize(frame, (width, height + 32 - (height % 32)))

            frame_bytes = cv2.imencode(".png", frame)[1].tobytes()
            # self.generate_diffused_frame(frame_bytes)
            self.generate_diffused_frame(frame_bytes)
            loop_count += 1
            self.progress_signal.emit(frame_count / video_frames_count * 100)
            print("progress: " + str(frame_count / video_frames_count * 100))
            print("test")
            # show preview of 4 generated diffused frames
            generation_count += 1
            if generation_count == 2:  # if last frame // video_frames_count
                generation_count = 0
                print("Generating preview...")

                writer = imageio.get_writer("test2.mp4", fps=30)
                try:
                    for frame in self.generated_preview_images:
                        # convert scalar image to numpy array
                        frame = np.array(frame)

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        writer.append_data(frame)

                    writer.close()
                    print("Preview generated!")

                    print("running signal")

                    # set video duration

                    if is_preview == False:
                        is_preview = True
                        self.preview_signal.emit(1)

                except Exception as e:
                    print(e)

        cap.release()
        print(f"{frame_count} frames extracted and saved to {frames_dir}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Uploader")
        self.setFixedSize(900, 700)

        # Create the main layout
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        # set color of main layout
        main_widget.setStyleSheet("background-color: #FFFFF2;")

        # Create prompt input

        # create QT input
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt here")
        self.prompt_input.textChanged.connect(self.set_prompt)
        main_layout.addWidget(self.prompt_input)

        # create negative prompt input
        self.negative_prompt_input = QLineEdit()
        self.negative_prompt_input.setPlaceholderText("Enter negative prompt here")
        self.negative_prompt_input.textChanged.connect(self.set_negative_prompt)
        main_layout.addWidget(self.negative_prompt_input)

        # style prompt input
        self.prompt_input.setStyleSheet(
            "background-color: #FFFFFF; border: 1px solid black; border-radius: 5px; padding: 5px;"
        )
        self.negative_prompt_input.setStyleSheet(
            "background-color: #FFFFFF; border: 1px solid black; border-radius: 5px; padding: 5px;"
        )

        # Create the options
        options_layout = QHBoxLayout()
        # set height of options layout

        # Create the label for the options
        options_label = QLabel("Options:")
        options_label.setStyleSheet("font-weight: bold;")

        # create checkbox for preview
        self.preview_checkbox = QCheckBox("Preview")
        self.preview_checkbox.setChecked(True)

        options_layout.addWidget(options_label)
        options_layout.addWidget(self.preview_checkbox)
        main_layout.addLayout(options_layout)

        # Create the upload button and its label
        upload_button = QPushButton("Upload Image", self)
        upload_label = QLabel("Upload an image file for face replacement (optional)")

        # Create the open button and its label
        open_button = QPushButton("Open Video File", self)
        open_label = QLabel("Upload a video file to be processed")

        # Create the layout for the upload button and label
        upload_layout = QVBoxLayout()
        upload_layout.addWidget(upload_button)
        upload_layout.addWidget(upload_label)

        # style the upload button

        upload_button.setFixedHeight(50)
        upload_button.setFixedWidth(200)
        upload_button.clicked.connect(self.upload_image)

        buttons_stlesheet = """
            QPushButton {
                background-color: #41D3BD;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                border: 2px solid #41D3BD;
            }
            QPushButton:hover {
                background-color : #1E90FF;
                color: white;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:pressed {
                background-color : #1E90FF;
            }
        """

        upload_button.setStyleSheet(buttons_stlesheet)

        # style the open button
        open_button.setFixedHeight(50)
        open_button.setFixedWidth(200)
        open_button.setStyleSheet(
            "background-color: #1E90FF; color: white; font-weight: bold; border-radius: 10px; border: 2px solid #1E90FF;"
        )
        # style hover
        open_button.setStyleSheet(buttons_stlesheet)
        open_button.clicked.connect(self.open_file_dialog)

        # Create the layout for the open button and label
        open_layout = QVBoxLayout()
        open_layout.addWidget(open_button)
        open_layout.addWidget(open_label)

        # Add the button and label layouts to the main layout
        button_layout = QHBoxLayout()
        button_layout.addLayout(upload_layout)
        button_layout.addLayout(open_layout)
        button_layout
        main_layout.addLayout(button_layout)

        # Set up the video player
        self.mediaPlayer = QMediaPlayer(parent=None)
        self.mediaPlayer.source  # property of type QUrl

        self.video_widget = QVideoWidget()
        self.video_widget.setFixedHeight(512)

        self.mediaPlayer.setVideoOutput(self.video_widget)

        main_layout.addWidget(self.video_widget)

        # Create the progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # style the progress bar
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(main_widget)
        self.setWindowTitle("Video Frame Extractor")

        self.video_file_path = None

    def handle_preview_data(self):
        print("Handling preview data...")
        video_path = "test2.mp4"
        try:
            qurl = QUrl.fromLocalFile("test2.mp4")

            # run self.handle_media_status_changed when media end

            print(str(self.mediaPlayer.mediaStatus()))
            if self.mediaPlayer.mediaStatus() == QMediaPlayer.MediaStatus.NoMedia:
                print("no media loaded")
                self.mediaPlayer.setSource(qurl)
                self.mediaPlayer.play()

            print("Preview data handled!")

        except Exception as e:
            print(e)

    def handle_media_status_changed(self, status):
        # if video is finished, play again
        if status == self.mediaPlayer.MediaStatus.EndOfMedia:
            print(status)
            print("End of media reached, playing again")

            # clear media player cache
            self.mediaPlayer.stop()
            # TODO: test video if it is finished
            if is_video_correct("test2.mp4"):
                self.mediaPlayer.setSource(
                    (QUrl.fromLocalFile("images/ExpandingUniverse2.mp4"))
                )
                self.mediaPlayer.setSource(QUrl.fromLocalFile("test2.mp4"))
                self.mediaPlayer.setPosition(0)
                self.mediaPlayer.play()
            else:
                self.handle_media_status_changed(
                    self.mediaPlayer.MediaStatus.EndOfMedia
                )

    def set_prompt(self, text):
        self.prompt_input = text

        # print("from global:" + str(prompt))

    def set_negative_prompt(self, text):
        self.negative_prompt_input = text

        # print("from global:" + str(negative_prompt))

    def open_file_dialog(self):
        # stop media player
        self.mediaPlayer.stop()
        print("Opening file dialog...")
        self.video_file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi)"
        )
        print(f"Selected video file: {self.video_file_path}")
        self.process_video_file()

    def upload_image(self):
        # Open a file dialog for selecting an image file
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)"
        )

        # TODO: Process the selected image file

    def process_video_file(self):
        if not self.video_file_path:
            return

        # Disable the open button and show the progress bar

        self.progress_bar.setValue(0)
        self.progress_bar.show()

        # Create and start the video processing thread
        self.video_thread = VideoProcessorThread(self.video_file_path)

        # Connect the signals to the slots
        self.video_thread.progress_signal.connect(self.update_progress_bar)
        self.video_thread.preview_signal.connect(self.handle_preview_data)
        self.video_thread.set_prompts(self.prompt_input, self.negative_prompt_input)
        self.mediaPlayer.mediaStatusChanged.connect(self.handle_media_status_changed)

        self.video_thread.start()

    def update_progress_bar(self, frame_count):
        # Update the progress bar
        self.progress_bar.setValue(frame_count)

        # If all frames have been processed, enable the open button and hide the progress bar
        if frame_count == self.progress_bar.maximum():
            self.open_button.setDisabled(False)
            # self.progress_bar.hide()


def main():
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    main()
