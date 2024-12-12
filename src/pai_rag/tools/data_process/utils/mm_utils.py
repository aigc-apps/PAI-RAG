# Others
def size_to_bytes(size):
    alphabets_list = [char for char in size if char.isalpha()]
    numbers_list = [char for char in size if char.isdigit()]

    if len(numbers_list) == 0:
        raise ValueError(f"Your input `size` does not contain numbers: {size}")

    size_numbers = int(float("".join(numbers_list)))

    if len(alphabets_list) == 0:
        # by default, if users do not specify the units, the number will be
        # regarded as in bytes
        return size_numbers

    suffix = "".join(alphabets_list).lower()

    if suffix == "kb" or suffix == "kib":
        return size_numbers << 10
    elif suffix == "mb" or suffix == "mib":
        return size_numbers << 20
    elif suffix == "gb" or suffix == "gib":
        return size_numbers << 30
    elif suffix == "tb" or suffix == "tib":
        return size_numbers << 40
    elif suffix == "pb" or suffix == "pib":
        return size_numbers << 50
    elif suffix == "eb" or suffix == "eib":
        return size_numbers << 60
    elif suffix == "zb" or suffix == "zib":
        return size_numbers << 70
    elif suffix == "yb" or suffix == "yib":
        return size_numbers << 80
    else:
        raise ValueError(
            f"You specified unidentifiable unit: {suffix}, "
            f"expected in [KB, MB, GB, TB, PB, EB, ZB, YB, "
            f"KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB], "
            f"(case insensitive, counted by *Bytes*)."
        )
