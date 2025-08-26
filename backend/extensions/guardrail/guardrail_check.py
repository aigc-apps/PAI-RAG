from pydantic import BaseModel
from extensions.guardrail.config import (
    CHECK_INPUT_ALGO,
    CHECK_OUTPUT_AGLO,
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


class GuardrailChecker:
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        region_id: str,
        endpoint: str
    ):
        aliyun_config = Config(
            # 阿里云账号AccessKey拥有所有API的访问权限，建议您使用RAM用户进行API访问或日常运维。
            # 强烈建议不要把AccessKey ID和AccessKey Secret保存到工程代码里，否则可能导致AccessKey泄露，威胁您账号下所有资源的安全。
            # 常见获取环境变量方式：
            # 获取RAM用户AccessKey ID：os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
            # 获取RAM用户AccessKey Secret：os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            # 连接超时时间 单位毫秒(ms)
            connect_timeout=10000,
            # 读超时时间 单位毫秒(ms)
            read_timeout=3000,
            region_id=region_id,
            endpoint=endpoint,
        )

        self.client = Client(aliyun_config)


    async def _acheck(self, text: str, check_algo: str = CHECK_INPUT_ALGO):
        if not text:
            return TextCheckResult(
                reject=False,
                reason="Empty text.",
                risk_level="low",
            )

        start = time.time()

        serviceParameters = {"content": text}

        textModerationPlusRequest = models.TextModerationPlusRequest(
            # 检测类型
            service=check_algo,
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

                advice = None
                if reject and len(response.body.data.advice) > 0:
                    advice = response.body.data.advice[0].answer

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
                    f"Check text {text} success. result:{response}. Elaspsed: {time.time() - start} seconds."
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
            )


    async def acheck_input(self, text):
        logger.info(f"Checking input: {text}.")
        return await self._acheck(text, check_algo=CHECK_INPUT_ALGO)

    async def acheck_output(self, text, current_result: TextCheckResult):
        if current_result.reject:
            return

        logger.info(f"Checking output: {text}.")

        new_check_result = await self._acheck(text, check_algo=CHECK_OUTPUT_AGLO)
        if new_check_result.reject:
            current_result.reject = new_check_result.reject
            current_result.advice = new_check_result.advice
            logger.info(f"Guardrail check failed for streaming output {text}. Advice: {current_result.advice}.")

        return
