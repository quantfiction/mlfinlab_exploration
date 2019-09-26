def returns_standard(entry_price, exit_price, fee):
    return (exit_price/entry_price)*(1-fee) - fee - 1


def returns_inverse(entry_price, exit_price, fee):
    return (-entry_price/exit_price)*(1 + fee) - fee + 1
