from flask import Flask, jsonify, render_template, request, send_file
import io
import boto3
from PIL import Image
import numpy as np

application = app = Flask(__name__)

dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('<YOUR_TABLE_NAME>')  # Replace with your DynamoDB table name

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_image', methods=['GET'])
def get_image():
    image_id = request.args.get('image_id')
    step_num = request.args.get('step_num')
    black_image = Image.open("static/images/" + image_id + "_female.png")
    white_image = Image.open("static/images/" + image_id + "_male.png")
    image1_np = np.array(black_image, dtype=np.float32)
    image2_np = np.array(white_image, dtype=np.float32)

    # Function to interpolate between two images
    alpha = int(step_num) / 20.0
    interpolated_image = (1 - alpha) * image1_np + alpha * image2_np
    interpolated_image = Image.fromarray(interpolated_image.astype('uint8'))

    img_io = io.BytesIO()
    interpolated_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

@app.route('/set1')
def page1():
    return render_template('app.html')

@app.route('/data', methods=['GET'])
def data():
    response = table.scan()
    items = response['Items']
    data = {}
    for item in items:
        data[item['image_id']] = {
                                    'A_realistic': str(item['A_realistic']),
                                    'B_realistic': str(item['A_realistic']),
                                    'ab_diff': str(item['ab_diff']),
                                    'ba_diff': str(item['ba_diff'])
                                }
    
    # print(data)
    return jsonify(data)

@app.route('/submit', methods=['POST'])
def handle_data():
    req_data = request.json  # This is the dictionary data sent from the client
    # response = table.scan()
    # orig_data = response['Items']
    # orig_data_dict = {}
    # for item in orig_data:
    #     orig_data_dict[item['image_id']] = {
    #                                 'A_realistic': str(item['A_realistic']),
    #                                 'B_realistic': str(item['A_realistic']),
    #                                 'ab_diff': str(item['ab_diff']),
    #                                 'ba_diff': str(item['ba_diff'])
    #                             }
    # print(req_data)
    for key in req_data.keys():
        # if key not in orig_data_dict:
        item = {}
        item = req_data[key]
        item['image_id'] = key
        response = table.put_item(Item=item)
    # Process your data here
    return jsonify({'status': 'success', 'message': 'Data received'})

if __name__ == '__main__':
    app.run(debug=False, port=8000)