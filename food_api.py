from flask import Flask, request, abort, jsonify, flash
import os
from pprint import pprint
from flask import Flask
from flask_limiter import Limiter
from PIL import Image


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    print('\n\n', width, height, '\n\n')
    return width, height
def check_pixels_size(filepath):
    width, height = get_num_pixels(filepath)
    if width < 544 and height < 544:
        return True
    else:
        return False
def get_user_key():
    return request.headers.get('user_key')

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
limiter = Limiter(app, key_func=get_user_key)

@app.route('/anpr/v1', methods = ['POST'])
@limiter.limit("1/minute")
def handle_image():
    pprint(request)
    if request.method == 'POST':
        key = get_user_key()
        if key == '110' or key == '100':
            imgpath = request.form.get['image']

            if imgpath and allowed_file(imgpath):
                if check_pixels_size(imgpath):
                    return jsonify({'number': '1234a-b', 'confidence': 70})
                return 'File size too big\n'
            else:
                abort(401)
        else:
            abort(400)
    else:
        abort(403)



if (__name__ == "__main__"):
    try:
        app.run()
    except:
        abort(403)
