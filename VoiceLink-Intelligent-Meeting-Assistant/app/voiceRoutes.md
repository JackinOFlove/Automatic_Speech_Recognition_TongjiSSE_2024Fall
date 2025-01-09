# 后端接口

# 开始会议 用户输入语言和角色分离设置来开始会议（角色可以设置0-10人）  语言可以设置（支持cn（中文）、en（英文）、粤语（yue）、日语（ja）、韩语（ko）。也可以设置multilingual 自动监测语言）
# 开始会议启动时和后端连接一个独立的websocket 用来推流分离结果

# 会议过程中 从会议开始开始计时 每5min进行一次over操作 并在over后对服务端进行轮询 直到status为completed
# response结果如下：
response:
{
    "Code": "0",
    "Data": {
        "TaskId": "fa2c9c6650a040699e26d0dd8642f15b",
        "TaskKey": "task20241207183903",
        "TaskStatus": "COMPLETED",
        "OutputMp3Path": "https://prod-tingwu-paas-common-beijing.oss-cn-beijing.aliyuncs.com/tingwu/output/1897412986514575/fa2c9c6650a040699e26d0dd8642f15b/fa2c9c6650a040699e26d0dd8642f15b_20241207183902.mp3?Expires=1736160430&OSSAccessKeyId=LTAI5tMzZ1D4o1drkJN1TfCr&Signature=T8Ak7ZqoXZ9Yc0AGyzgukOjFe7c%3D",
        "Result": {
            "Transcription": "https://prod-tingwu-paas-common-beijing.oss-cn-beijing.aliyuncs.com/tingwu/output/1897412986514575/fa2c9c6650a040699e26d0dd8642f15b/fa2c9c6650a040699e26d0dd8642f15b_Transcription_20241207184700.json?Expires=1736160430&OSSAccessKeyId=LTAI5tMzZ1D4o1drkJN1TfCr&Signature=fjLEG7X%2Bv40YJbwfXuKNlDt0Urk%3D"
        }
    },
    "Message": "success",
    "RequestId": "97BEE520-563D-5601-83F1-B92E1F7DAE60"
}


# 得到结果后 到transcription字段获取文件链接
# 对文件进行解析 并通过独立的websocket推流给前端
# 文件的结构如下：
{
    "TaskId": "fa2c9c6650a040699e26d0dd8642f15b",
    "Transcription": {
        "AudioInfo": {
            "Duration": 118916
        },
        "Paragraphs": [
            {
                "ParagraphId": "1733568420851500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 1,
                        "SentenceId": 1,
                        "Start": 80,
                        "End": 700,
                        "Text": "你好，"
                    },
                    {
                        "Id": 2,
                        "SentenceId": 1,
                        "Start": 700,
                        "End": 986,
                        "Text": "现在"
                    },
                    {
                        "Id": 3,
                        "SentenceId": 1,
                        "Start": 986,
                        "End": 1129,
                        "Text": "还"
                    },
                    {
                        "Id": 4,
                        "SentenceId": 1,
                        "Start": 1129,
                        "End": 1272,
                        "Text": "能"
                    },
                    {
                        "Id": 5,
                        "SentenceId": 1,
                        "Start": 1272,
                        "End": 1415,
                        "Text": "用"
                    },
                    {
                        "Id": 6,
                        "SentenceId": 1,
                        "Start": 1415,
                        "End": 1558,
                        "Text": "吗？"
                    },
                    {
                        "Id": 7,
                        "SentenceId": 1,
                        "Start": 4880,
                        "End": 5415,
                        "Text": "This "
                    },
                    {
                        "Id": 8,
                        "SentenceId": 1,
                        "Start": 5415,
                        "End": 6490,
                        "Text": "suggestion. "
                    }
                ]
            },
            {
                "ParagraphId": "1733568448961500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 9,
                        "SentenceId": 2,
                        "Start": 28190,
                        "End": 28714,
                        "Text": "测试"
                    },
                    {
                        "Id": 10,
                        "SentenceId": 2,
                        "Start": 28714,
                        "End": 29240,
                        "Text": "开始。"
                    }
                ]
            },
            {
                "ParagraphId": "1733568462201500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 11,
                        "SentenceId": 3,
                        "Start": 41430,
                        "End": 42300,
                        "Text": "suggestion. "
                    }
                ]
            },
            {
                "ParagraphId": "1733568479211500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 12,
                        "SentenceId": 4,
                        "Start": 58440,
                        "End": 58628,
                        "Text": "识"
                    },
                    {
                        "Id": 13,
                        "SentenceId": 4,
                        "Start": 58628,
                        "End": 59004,
                        "Text": "别的"
                    },
                    {
                        "Id": 14,
                        "SentenceId": 4,
                        "Start": 59004,
                        "End": 59192,
                        "Text": "还"
                    },
                    {
                        "Id": 15,
                        "SentenceId": 4,
                        "Start": 59192,
                        "End": 59568,
                        "Text": "可以"
                    },
                    {
                        "Id": 16,
                        "SentenceId": 4,
                        "Start": 59568,
                        "End": 59760,
                        "Text": "吧。"
                    }
                ]
            },
            {
                "ParagraphId": "1733568489611500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 17,
                        "SentenceId": 5,
                        "Start": 68840,
                        "End": 69790,
                        "Text": "可以。"
                    }
                ]
            },
            {
                "ParagraphId": "1733568498227500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 18,
                        "SentenceId": 6,
                        "Start": 77456,
                        "End": 77613,
                        "Text": "不"
                    },
                    {
                        "Id": 19,
                        "SentenceId": 6,
                        "Start": 77613,
                        "End": 77927,
                        "Text": "知道"
                    },
                    {
                        "Id": 20,
                        "SentenceId": 6,
                        "Start": 77927,
                        "End": 78241,
                        "Text": "刚才"
                    },
                    {
                        "Id": 21,
                        "SentenceId": 6,
                        "Start": 78241,
                        "End": 78712,
                        "Text": "有没有"
                    },
                    {
                        "Id": 22,
                        "SentenceId": 6,
                        "Start": 78712,
                        "End": 79026,
                        "Text": "处理"
                    },
                    {
                        "Id": 23,
                        "SentenceId": 6,
                        "Start": 79026,
                        "End": 79340,
                        "Text": "完毕，"
                    },
                    {
                        "Id": 24,
                        "SentenceId": 6,
                        "Start": 80596,
                        "End": 80924,
                        "Text": "第二"
                    },
                    {
                        "Id": 25,
                        "SentenceId": 6,
                        "Start": 80924,
                        "End": 81088,
                        "Text": "次"
                    },
                    {
                        "Id": 26,
                        "SentenceId": 6,
                        "Start": 81088,
                        "End": 81416,
                        "Text": "测试"
                    },
                    {
                        "Id": 27,
                        "SentenceId": 6,
                        "Start": 81416,
                        "End": 81746,
                        "Text": "开始。"
                    }
                ]
            },
            {
                "ParagraphId": "1733568510774500000",
                "SpeakerId": "1",
                "Words": [
                    {
                        "Id": 28,
                        "SentenceId": 7,
                        "Start": 90003,
                        "End": 90305,
                        "Text": "那个"
                    },
                    {
                        "Id": 29,
                        "SentenceId": 7,
                        "Start": 90305,
                        "End": 90607,
                        "Text": "目的"
                    },
                    {
                        "Id": 30,
                        "SentenceId": 7,
                        "Start": 90607,
                        "End": 90758,
                        "Text": "是"
                    },
                    {
                        "Id": 31,
                        "SentenceId": 7,
                        "Start": 90758,
                        "End": 90909,
                        "Text": "他"
                    },
                    {
                        "Id": 32,
                        "SentenceId": 7,
                        "Start": 90909,
                        "End": 91211,
                        "Text": "这个"
                    },
                    {
                        "Id": 33,
                        "SentenceId": 7,
                        "Start": 91211,
                        "End": 91362,
                        "Text": "是"
                    },
                    {
                        "Id": 34,
                        "SentenceId": 7,
                        "Start": 91362,
                        "End": 91513,
                        "Text": "吧？"
                    },
                    {
                        "Id": 35,
                        "SentenceId": 7,
                        "Start": 91783,
                        "End": 91919,
                        "Text": "对"
                    },
                    {
                        "Id": 36,
                        "SentenceId": 7,
                        "Start": 91919,
                        "End": 92055,
                        "Text": "对"
                    },
                    {
                        "Id": 37,
                        "SentenceId": 7,
                        "Start": 92055,
                        "End": 92191,
                        "Text": "对，"
                    },
                    {
                        "Id": 38,
                        "SentenceId": 7,
                        "Start": 92193,
                        "End": 92455,
                        "Text": "就是"
                    },
                    {
                        "Id": 39,
                        "SentenceId": 7,
                        "Start": 92455,
                        "End": 92586,
                        "Text": "我"
                    },
                    {
                        "Id": 40,
                        "SentenceId": 7,
                        "Start": 92586,
                        "End": 92848,
                        "Text": "这个"
                    },
                    {
                        "Id": 41,
                        "SentenceId": 7,
                        "Start": 92848,
                        "End": 93110,
                        "Text": "就是"
                    },
                    {
                        "Id": 42,
                        "SentenceId": 7,
                        "Start": 93110,
                        "End": 93241,
                        "Text": "我"
                    },
                    {
                        "Id": 43,
                        "SentenceId": 7,
                        "Start": 93241,
                        "End": 93503,
                        "Text": "这个"
                    },
                    {
                        "Id": 44,
                        "SentenceId": 7,
                        "Start": 93503,
                        "End": 93634,
                        "Text": "他"
                    },
                    {
                        "Id": 45,
                        "SentenceId": 7,
                        "Start": 93634,
                        "End": 93896,
                        "Text": "肯定"
                    },
                    {
                        "Id": 46,
                        "SentenceId": 7,
                        "Start": 93896,
                        "End": 94289,
                        "Text": "打印机"
                    },
                    {
                        "Id": 47,
                        "SentenceId": 7,
                        "Start": 94289,
                        "End": 94420,
                        "Text": "把"
                    },
                    {
                        "Id": 48,
                        "SentenceId": 7,
                        "Start": 94420,
                        "End": 94551,
                        "Text": "它"
                    },
                    {
                        "Id": 49,
                        "SentenceId": 7,
                        "Start": 94551,
                        "End": 94682,
                        "Text": "接"
                    },
                    {
                        "Id": 50,
                        "SentenceId": 7,
                        "Start": 94682,
                        "End": 94944,
                        "Text": "进去"
                    },
                    {
                        "Id": 51,
                        "SentenceId": 7,
                        "Start": 94944,
                        "End": 95075,
                        "Text": "就"
                    },
                    {
                        "Id": 52,
                        "SentenceId": 7,
                        "Start": 95075,
                        "End": 95206,
                        "Text": "行"
                    },
                    {
                        "Id": 53,
                        "SentenceId": 7,
                        "Start": 95206,
                        "End": 95343,
                        "Text": "了。"
                    },
                    {
                        "Id": 54,
                        "SentenceId": 8,
                        "Start": 95343,
                        "End": 96109,
                        "Text": "因为"
                    },
                    {
                        "Id": 55,
                        "SentenceId": 8,
                        "Start": 96109,
                        "End": 96545,
                        "Text": "这个"
                    },
                    {
                        "Id": 56,
                        "SentenceId": 8,
                        "Start": 96545,
                        "End": 96763,
                        "Text": "我"
                    },
                    {
                        "Id": 57,
                        "SentenceId": 8,
                        "Start": 96763,
                        "End": 97199,
                        "Text": "感觉"
                    },
                    {
                        "Id": 58,
                        "SentenceId": 8,
                        "Start": 97199,
                        "End": 97635,
                        "Text": "这个"
                    },
                    {
                        "Id": 59,
                        "SentenceId": 8,
                        "Start": 97635,
                        "End": 98071,
                        "Text": "然后"
                    },
                    {
                        "Id": 60,
                        "SentenceId": 8,
                        "Start": 98071,
                        "End": 98289,
                        "Text": "我"
                    },
                    {
                        "Id": 61,
                        "SentenceId": 8,
                        "Start": 98289,
                        "End": 98725,
                        "Text": "主页"
                    },
                    {
                        "Id": 62,
                        "SentenceId": 8,
                        "Start": 98725,
                        "End": 98943,
                        "Text": "我"
                    },
                    {
                        "Id": 63,
                        "SentenceId": 8,
                        "Start": 98943,
                        "End": 99161,
                        "Text": "也"
                    },
                    {
                        "Id": 64,
                        "SentenceId": 8,
                        "Start": 99161,
                        "End": 99597,
                        "Text": "特别"
                    },
                    {
                        "Id": 65,
                        "SentenceId": 8,
                        "Start": 99597,
                        "End": 100033,
                        "Text": "方便"
                    },
                    {
                        "Id": 66,
                        "SentenceId": 8,
                        "Start": 100033,
                        "End": 100251,
                        "Text": "发，"
                    },
                    {
                        "Id": 67,
                        "SentenceId": 8,
                        "Start": 102153,
                        "End": 102549,
                        "Text": "为什么"
                    },
                    {
                        "Id": 68,
                        "SentenceId": 8,
                        "Start": 102549,
                        "End": 102813,
                        "Text": "如此"
                    },
                    {
                        "Id": 69,
                        "SentenceId": 8,
                        "Start": 102813,
                        "End": 103083,
                        "Text": "开心？"
                    },
                    {
                        "Id": 70,
                        "SentenceId": 9,
                        "Start": 105273,
                        "End": 105503,
                        "Text": "对，"
                    },
                    {
                        "Id": 71,
                        "SentenceId": 9,
                        "Start": 105503,
                        "End": 106899,
                        "Text": "但是"
                    },
                    {
                        "Id": 72,
                        "SentenceId": 9,
                        "Start": 106899,
                        "End": 108295,
                        "Text": "打开"
                    },
                    {
                        "Id": 73,
                        "SentenceId": 9,
                        "Start": 108295,
                        "End": 109691,
                        "Text": "分析，"
                    },
                    {
                        "Id": 74,
                        "SentenceId": 9,
                        "Start": 114273,
                        "End": 114547,
                        "Text": "就是"
                    },
                    {
                        "Id": 75,
                        "SentenceId": 9,
                        "Start": 114547,
                        "End": 114823,
                        "Text": "这个。"
                    }
                ]
            }
        ],
        "AudioSegments": [
            [
                80,
                6490
            ],
            [
                28190,
                29240
            ],
            [
                41430,
                42300
            ],
            [
                58440,
                59760
            ],
            [
                68840,
                69790
            ],
            [
                77456,
                81746
            ],
            [
                90003,
                95343
            ],
            [
                95343,
                103083
            ],
            [
                105273,
                114823
            ]
        ]
    }
}

# 会议结束

