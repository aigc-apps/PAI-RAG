from pydantic import BaseModel
from pairag.integrations.guardrail.config import (
    DEFAULT_GUARDRAIL_ADVICE,
    AliyunTextModerationPlusConfig,
)
from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
import json
import time
from loguru import logger


class TextCheckResult(BaseModel):
    reject: bool = False  # 是否拒绝
    reason: str | None = None
    risk_level: str = "low"
    advice: str | None = None


class PaiLlmGuardrail:
    def __init__(self, config: AliyunTextModerationPlusConfig):
        aliyun_config = Config(
            # 阿里云账号AccessKey拥有所有API的访问权限，建议您使用RAM用户进行API访问或日常运维。
            # 强烈建议不要把AccessKey ID和AccessKey Secret保存到工程代码里，否则可能导致AccessKey泄露，威胁您账号下所有资源的安全。
            # 常见获取环境变量方式：
            # 获取RAM用户AccessKey ID：os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
            # 获取RAM用户AccessKey Secret：os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
            access_key_id=config.access_key_id,
            access_key_secret=config.access_key_secret,
            # 连接超时时间 单位毫秒(ms)
            connect_timeout=10000,
            # 读超时时间 单位毫秒(ms)
            read_timeout=3000,
            region_id=config.region,
            endpoint=config.endpoint,
        )

        self.custom_advice = config.custom_advice
        self.client = Client(aliyun_config)

    async def acheck(self, text):
        start = time.time()

        serviceParameters = {"content": text}

        textModerationPlusRequest = models.TextModerationPlusRequest(
            # 检测类型
            service="query_security_check",
            service_parameters=json.dumps(serviceParameters),
        )

        try:
            response = await self.client.text_moderation_plus_async(
                textModerationPlusRequest
            )
            if response.status_code == 200 and response.body.code == 200:
                # 调用成功
                risk_level = response.body.data.risk_level
                reject = False
                if risk_level.lower() == "high":
                    reject = True

                advice = self.custom_advice
                if not advice and reject:
                    if len(response.body.data.advice) > 0:
                        advice = response.body.data.advice[0].answer
                    else:
                        advice = DEFAULT_GUARDRAIL_ADVICE

                reason = None
                if reject and len(response.body.data.result) > 0:
                    reason = response.body.data.result[0].description

                result = TextCheckResult(
                    reject=reject,
                    reason=reason,
                    risk_level=response.body.data.risk_level,
                    advice=advice,
                )

                logger.info(
                    f"Check text {text} success. result:{result}. Elaspsed: {time.time() - start} seconds."
                )
                return result
            else:
                logger.info(
                    f"Check text response failed. status:{response.status_code} ,result:{response}, Elaspsed: {time.time() - start} seconds."
                )
                return TextCheckResult(
                    reject=False,
                    reason="request failed",
                    risk_level="unknown",
                    advice="internal error",
                )
        except Exception as err:
            logger.info(
                f"Unhandled error: check text failed due to {err}. Elaspsed: {time.time() - start} seconds."
            )
            return TextCheckResult(
                reject=False,
                reason="Check text failed.",
                risk_level="low",
                advice="",
            )
