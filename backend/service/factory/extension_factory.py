from extensions.guardrail.guardrail_check import GuardrailChecker
from common.encrypt_utils import decrypt_key
from db.models.guardrail import GuardrailConfigEntity



def create_guardrail_checker(config: GuardrailConfigEntity) -> GuardrailChecker:
    if not config or not config.encrypted_access_key_id or not config.encrypted_access_key_secret or not config.region_id or not config.endpoint:
        raise ValueError("Guardrail config is incomplete.")

    return GuardrailChecker(
        access_key_id=decrypt_key(config.encrypted_access_key_id),
        access_key_secret=decrypt_key(config.encrypted_access_key_secret),
        region_id=config.region_id,
        endpoint=config.endpoint,
    )
