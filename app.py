from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow import keras
from werkzeug.utils import secure_filename

from flask import send_file

app = Flask(__name__)

MODEL_PATH = 'full_CNN_model.h5'
model = keras.models.load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

    def road_lines(self, frame):
        small_img = cv2.resize(frame, (160, 80))
        small_img = np.array(small_img)
        small_img = small_img[None, :, :, :]

        prediction = model.predict(small_img)[0] * 255
        self.recent_fit.append(prediction)

        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        self.avg_fit = np.mean(np.array([i for i in self.recent_fit]), axis=0)

        blanks = np.zeros_like(self.avg_fit).astype(np.uint8)
        lane_drawn = np.dstack((blanks, self.avg_fit, blanks))

        lane_image = cv2.resize(lane_drawn, (1280, 720))

        frame = frame.astype(np.uint8)
        lane_image = lane_image.astype(np.uint8)

        result = cv2.addWeighted(frame, 1, lane_image, 1, 0)

        return result

lanes = Lanes()

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mkv'}
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return False
    return True




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('error.html', error_message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('error.html', error_message='No file selected')



    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filename_no_ext, file_extension = os.path.splitext(filename)
            result_filename = f"result_{filename_no_ext}.mp4"
            result_path = os.path.normpath(os.path.join(app.config['RESULT_FOLDER'], result_filename)).replace(os.path.sep, '/')
            file.save(file_path)

            cap = cv2.VideoCapture(file_path)
            fourcc = cv2.VideoWriter_fourcc(*'H264')

            out = cv2.VideoWriter(result_path, fourcc, 20.0, (1280, 720))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = lanes.road_lines(frame)
                out.write(processed_frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            return render_template('preview.html', result_filename=result_filename)

        except cv2.error as e:
            # Handle OpenCV error - display an error page with the error message
            return render_template('error.html', error_message=f'Sorry you have uploaded an wrong video please check')

    else:
        return render_template('error.html', error_message='Invalid file format ')



@app.route('/results/<filename>')
def uploaded_file(filename):
    response = send_from_directory(app.config['RESULT_FOLDER'], filename, mimetype='video/mp4')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Inside your Flask app's route handling the preview
@app.route('/preview/<result_filename>')
def preview(result_filename):
    return render_template('preview.html', result_filename=result_filename)



if __name__ == '__main__':
    app.run(debug=True)
