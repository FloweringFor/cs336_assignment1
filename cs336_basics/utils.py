def bytes_to_unicode():
    """
    创建一个 0-255 字节到安全 Unicode 字符的映射表。
    """
    # 这些是 ASCII 中不需要转义的、安全的字符区间
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

