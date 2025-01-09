import csv
from flask import Blueprint, request, jsonify
from pathlib import Path
from datetime import datetime
import os

# create a Blueprint instance
meetingInfo_routes = Blueprint('meetingInfo', __name__, url_prefix='/meetingInfo')


# the route is http://127.0.0.1:5000/meetingInfo/saveMeeting
@meetingInfo_routes.route('/saveMeeting', methods=['POST'])
def save_meeting():
    # get data from request
    data = request.get_json()
    print("data",data)

    meeting_title = data.get('meetingTitle')

    summary = data.get('Summary')

    # check if required fields exist
    if not all([meeting_title, summary]):
        return jsonify({'error': '缺少必需的字段'}), 400

    # ensure directory exists
    data_dir = Path('app/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # use os.path.join to build file path
    file_path = os.path.join('app', 'data', 'meetingInfo.csv')

    # read file and find meeting_title
    updated = False
    rows = []
    if os.path.exists(file_path):
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['meetingTitle'] == meeting_title:  # use column name 'meetingTitle' to index
                    # find meeting_title, update summary
                    row['Summary'] = summary  # use column name 'Summary' to update
                    updated = True
                rows.append(row)

    if not updated:
        return jsonify({'error': '未找到指定的 meetingTitle'}), 404

    # write back to file
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['meetingID', 'meetingTitle', 'startTime', 'endTime', 'persons', 'Summary']  # ensure field names match CSV file
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return jsonify({'message': '会议信息已更新'}), 200


# the route is http://127.0.0.1:5000/meetingInfo/saveHalfMeeting
@meetingInfo_routes.route('/saveHalfMeeting', methods=['POST'])
def save_half_meeting():
    # get data from request
    data = request.get_json()
    meeting_title = data.get('meetingTitle')
    start_time = data.get('startTime')
    end_time = data.get('endTime')
    # convert timestamp to normal time format
    print("data=",data)
    #start_time = datetime.fromtimestamp(int(start_time) / 1000).strftime('%Y-%m-%dT%H:%M:%S')
    #end_time = datetime.fromtimestamp(int(end_time) / 1000).strftime('%Y-%m-%dT%H:%M:%S')
    persons = data.get('persons')

    # check if required fields exist
    if not all([meeting_title, start_time, end_time, persons]):
        return jsonify({'error': '缺少必需的字段'}), 400

    # ensure directory exists
    data_dir = Path('app/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # use os.path.join to build file path
    file_path = os.path.join('app', 'data', 'meetingInfo.csv')

    # get the largest meeting_id and add 1
    max_id = 0
    if os.path.exists(file_path):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0].isdigit():
                    max_id = max(max_id, int(row[0]))

    meeting_id = max_id + 1

    # write to file
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([meeting_id, meeting_title, start_time, end_time, persons, ""])  # ensure each line has an end character

    return jsonify({'message': '会议信息已保存'}), 201


# the route is http://127.0.0.1:5000/meetingInfo/getMeetings
@meetingInfo_routes.route('/getMeetings', methods=['GET'])
def get_meetings():
    # use os.path.join to build file path
    file_path = os.path.join('app', 'data', 'meetingInfo.csv')

    # read file
    meetings = []
    if os.path.exists(file_path):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                meetings.append(row)

    return jsonify(meetings), 200