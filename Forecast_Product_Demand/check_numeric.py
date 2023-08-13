def _check_if_numeric(value):
    """
    값이 만약 수치형이면 True, 아니면 에러대신 False 반환
    """
    try:
        pd.to_numeric(value)
        return True
    except (TypeError, ValueError):
        return False