import struct, os

def count_samples(idx_path):
    with open(idx_path, "rb") as f:
        f.seek(16)  # skip magic + version
        dtype = struct.unpack("B", f.read(1))[0]
        num_seq = struct.unpack("<q", f.read(8))[0]
    return num_seq

total = 0
for fname in os.listdir("dclm-tokenize"):
    if fname.endswith(".idx"):
        cnt = count_samples(os.path.join("dclm-tokenize", fname))
        total += cnt
        print(fname, cnt)
print("Total samples:", total)
