def safety_margin(loss_rate, window):
    S = 1.0 - (loss_rate * window)
    return {"S": float(S), "loss_rate": float(loss_rate), "window": float(window)}

