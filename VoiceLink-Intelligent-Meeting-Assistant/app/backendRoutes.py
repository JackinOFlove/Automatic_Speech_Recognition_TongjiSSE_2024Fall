
import json
from flask import Blueprint, request, jsonify
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from aliyunsdkcore.auth.credentials import AccessKeyCredential
import time
import requests
import datetime
import threading

# constant definition
ALIYUN_ACCESS_KEY = "xxx"
ALIYUN_SECRET_KEY = "xxx"
APP_KEY = "xxx"
REGION_ID = "cn-beijing"

# API configuration
API_CONFIG = {
    'domain': 'tingwu.cn-beijing.aliyuncs.com',
    'version': '2023-09-30',
    'protocol': 'https'
}

# supported language list
SUPPORTED_LANGUAGES = ['cn', 'en', 'yue', 'ja', 'ko', 'multilingual']

REQUEST_INTERVAL=10


# create blueprint
meeting_bp = Blueprint('meeting', __name__)

# store active meeting information
active_meetings = {}


def get_client():
    """get aliyun client"""
    credentials = AccessKeyCredential(ALIYUN_ACCESS_KEY, ALIYUN_SECRET_KEY)
    return AcsClient(region_id=REGION_ID, credential=credentials)


def create_common_request(domain, version, protocol_type, method, uri):
    """create common request"""
    ali_request = CommonRequest()
    ali_request.set_accept_format('json')
    ali_request.set_domain(domain)
    ali_request.set_version(version)
    ali_request.set_protocol_type(protocol_type)
    ali_request.set_method(method)
    ali_request.set_uri_pattern(uri)
    ali_request.add_header('Content-Type', 'application/json')
    return ali_request


def execute_stop_operation(task_id):
    """execute stop operation"""
    try:
        client = get_client()
        body = {
            'AppKey': APP_KEY,
            'Input': {'TaskId': task_id}
        }

        stop_request = create_common_request(
            API_CONFIG['domain'],
            API_CONFIG['version'],
            API_CONFIG['protocol'],
            'PUT',
            '/openapi/tingwu/v2/tasks'
        )
        stop_request.add_query_param('type', 'realtime')
        stop_request.add_query_param('operation', 'stop')
        stop_request.set_content(json.dumps(body).encode('utf-8'))

        response = client.do_action_with_exception(stop_request)
        print(f"Stop response: {response.decode('utf-8')}")
        return True
    except Exception as e:
        print(f"Stop operation error: {str(e)}")
        return False




def poll_meeting(task_id):
    """
    poll meeting status and execute over operation periodically
    returns: the last complete response
    """
    final_response = None
    try:
        if execute_stop_operation(task_id):
            while task_id in active_meetings:
                # get complete task information
                client = get_client()
                status_request = create_common_request(
                    API_CONFIG['domain'],
                    API_CONFIG['version'],
                    API_CONFIG['protocol'],
                    'GET',
                    f'/openapi/tingwu/v2/tasks/{task_id}'
                )

                response = client.do_action_with_exception(status_request)
                response_data = json.loads(response)
                status_result = response_data['Data']['TaskStatus']

                if not status_result:
                    time.sleep(REQUEST_INTERVAL)
                    continue

                print(f"Status response: {json.dumps(response_data, indent=4, ensure_ascii=False)}")

                # if completed
                if status_result == 'COMPLETED':
                    final_response = response_data
                    break
                time.sleep(REQUEST_INTERVAL)

    except Exception as e:
        print(f"Polling error for task {task_id}: {str(e)}")
    finally:
        print("final:",final_response)
        return final_response


def init_parameters(language='cn', speaker_count=2):
    print(speaker_count)
    """initialize meeting parameters"""
    body = {
        'AppKey': APP_KEY,
        'Input': {
            'Format': 'pcm',
            'SampleRate': 16000,
            'SourceLanguage': language,
            'TaskKey': 'task' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
            'ProgressiveCallbacksEnabled': False
        },
        'Parameters': {
            'Transcription': {
                'OutputLevel':2,
                'DiarizationEnabled': True,
                'Diarization': {
                    'SpeakerCount': speaker_count
                }
            }
        }
    }
    return body


@meeting_bp.route('/initMeetingInfo', methods=['POST'])
def start_meeting():
    """start meeting"""
    try:
        print("准备开始会议")
        data = request.json
        language = data.get('language', 'cn')
        speaker_count = int(data.get('speakerCount', 2))
        print("language:", language)
        print("speaker_count:", speaker_count)

        # 验证参数
        if speaker_count not in range(0, 11):
            return jsonify({"error": "Invalid speaker count"}), 400

        if language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": "Invalid language"}), 400

        body = init_parameters(language, speaker_count)
        print("body=", body)

        client = get_client()
        ali_request = create_common_request(
            API_CONFIG['domain'],
            API_CONFIG['version'],
            API_CONFIG['protocol'],
            'PUT',
            '/openapi/tingwu/v2/tasks'
        )
        ali_request.add_query_param('type', 'realtime')
        ali_request.set_content(json.dumps(body).encode('utf-8'))

        response = client.do_action_with_exception(ali_request)
        response_data = json.loads(response.decode('utf-8'))
        print(response_data)

        task_id = response_data['Data']['TaskId']
        ws_url = response_data['Data']['MeetingJoinUrl']
        active_meetings[task_id] = True


        # return formatted response
        return jsonify({
            "code": 200,
            "message": "success",
            "meetingId": task_id,
            "wsUrl": ws_url
        })

    except Exception as e:
        print("error:", e)
        return jsonify({"error": str(e)}), 500



@meeting_bp.route('/<task_id>/record', methods=['GET'])
def generate_meeting_record(task_id):
    print("task_id:",task_id)
    print("正在生成会议记录")
    active_meetings[task_id] = True
    """generate meeting record"""
    try:
        response_data = poll_meeting(task_id)

        if not response_data:
            return jsonify({"error": "获取任务状态失败"}), 500

        # return transcription result URL
        transcription_url = response_data['Data']['Result']['Transcription']
        print("url=",transcription_url)

        return jsonify({
            "status": "success",
            "transcription_url": transcription_url,
        }),200

    except Exception as e:
        print(f"Generate record error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@meeting_bp.route('/endMeeting', methods=['POST'])
def end_meeting():
    data = request.json
    task_id=data.get('taskId')
    print("task_id:",task_id)

    if active_meetings[task_id]:
        active_meetings[task_id] = False

    return jsonify({
        "code": 200,
        "message": "success",
    })

