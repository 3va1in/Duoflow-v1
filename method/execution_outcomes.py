"""跨官方 WS 包装保留 API 故障类别；模型格式和程序错误不按网络故障处理。"""
from __future__ import annotations

import json


API_ERROR_PREFIX = 'DUOFLOW_API_ERROR '
API_ERROR_CODES = frozenset({
    'network_error', 'stream_interrupted', 'transport_timeout', 'api_timeout',
    'rate_limit_exceeded', 'rate_limit_error', 'credit_balance_exhausted',
    'subscription_sharing_usage_limit_exceeded', 'insufficient_quota',
    'server_error', 'internal_server_error', 'service_unavailable', 'server_is_overloaded',
    'api_auth_error', 'api_http_error',
    'upstream_result_unknown', 'upstream_call_budget_exhausted',
    'existing_sharing_session_required',
})


class AstraPlannerError(RuntimeError):
    pass


class AstraAPIError(AstraPlannerError):
    def __init__(self, code, *, request_id=None, http_status=None):
        if code not in API_ERROR_CODES:
            raise ValueError('unclassified API error code')
        if request_id is not None and not isinstance(request_id, str):
            raise TypeError('request_id must be a string or null')
        if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
            raise ValueError('invalid HTTP status')
        self.payload = {'code': code, 'request_id': request_id, 'http_status': http_status}
        super().__init__(API_ERROR_PREFIX + json.dumps(self.payload, separators=(',', ':')))


class AstraTransportError(AstraAPIError):
    pass


def _remote_api_payload(error):
    text = str(error)
    if not text.startswith(API_ERROR_PREFIX):
        return None
    try:
        value = json.loads(text[len(API_ERROR_PREFIX):])
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, dict) or set(value) != {'code', 'request_id', 'http_status'}
            or value['code'] not in API_ERROR_CODES
            or value['request_id'] is not None and not isinstance(value['request_id'], str)
            or value['http_status'] is not None and (type(value['http_status']) is not int
                                                    or not 100 <= value['http_status'] <= 599)):
        return None
    return value


def classify_execution_error(error, *, phase):
    """仅在明确的策略 RPC 边界识别传输错误，不搜索宽泛错误文本。"""
    is_ws_error = (type(error).__name__ == 'WsError'
                   and type(error).__module__.startswith('client_server.ws.'))
    code = getattr(error, 'code', None)
    code = getattr(code, 'value', code)
    payload = error.payload if isinstance(error, AstraAPIError) else None
    if is_ws_error and code in {'call_failed', 'infer_failed'}:
        payload = _remote_api_payload(error)
    if payload is not None:
        return {'error_category': 'api_error', **payload}
    if phase == 'policy_rpc':
        if is_ws_error and code == 'timeout':
            return {'error_category': 'infrastructure_error', 'code': 'policy_rpc_timeout'}
        if isinstance(error, (TimeoutError, ConnectionError)):
            return {'error_category': 'infrastructure_error', 'code': 'policy_rpc_transport'}
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return {'error_category': 'infrastructure_error', 'code': 'process_interrupted'}
    return {'error_category': 'policy_error', 'code': 'policy_execution_error'}
