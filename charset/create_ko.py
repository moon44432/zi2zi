

with open("ko.json", "w", encoding="utf-8") as f:
    s = ""
    for idx in range(0xac00, 0xd7a4):
        s += ('"\\u' + "%04x" % idx + '", ')
    f.write(s)
