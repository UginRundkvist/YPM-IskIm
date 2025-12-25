import random
from typing import List, Tuple

# Датасет: (x0, x1, x2, y)
data: List[Tuple[float, float, float, int]] = [
    (1.0, -0.15,  0.20, 0),
    (1.0, -1.20, -0.55, 0),
    (1.0, -0.85,  1.00, 0),
    (1.0, -0.35,  1.20, 0),
    (1.0, -1.10,  0.50, 0),
    (1.0,  0.70,  0.70, 1),
    (1.0,  1.50,  1.40, 1),
    (1.0,  0.80, -0.50, 1),
    (1.0, -0.80, -1.20, 1),
    (1.0,  0.75, -1.90, 1),
]

def predict(w: Tuple[float, float, float], x: Tuple[float, float, float]) -> int:
    z = w[0]*x[0] + w[1]*x[1] + w[2]*x[2]
    return 1 if z >= 0 else 0  # порог

def hebb_train_until_converge(
    w0: Tuple[float, float, float],
    max_epochs: int = 1000,
    log: bool = False,
):
    w = list(w0)
    epochs = 0
    history = []
    while epochs < max_epochs:
        changed = False
        epoch_log = []
        for x0, x1, x2, y in data:
            x = (x0, x1, x2)
            yhat = predict(tuple(w), x)
            if yhat == y:
                epoch_log.append((x, y, yhat, tuple(w), "ok"))
                continue
            changed = True
            if yhat == 0 and y == 1:
                # увеличить веса: w += x
                w[0] += x0; w[1] += x1; w[2] += x2
                epoch_log.append((x, y, yhat, tuple(w), "w += x"))
            elif yhat == 1 and y == 0:
                # уменьшить веса: w -= x
                w[0] -= x0; w[1] -= x1; w[2] -= x2
                epoch_log.append((x, y, yhat, tuple(w), "w -= x"))
        epochs += 1
        history.append(epoch_log)
        if not changed:
            break
    if log:
        return tuple(w), epochs, history
    return tuple(w), epochs

def try_find_initial_weights_for_exact_epochs(
    target_epochs: int = 3,
    trials: int = 20000,
    grid: bool = False,
):
    if grid:
        # Сеточный перебор по [-1, 1] с шагом 0.5 (или 0.25 — по желанию)
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for w0 in vals:
            for w1 in vals:
                for w2 in vals:
                    _, ep = hebb_train_until_converge((w0, w1, w2))
                    if ep == target_epochs:
                        return (w0, w1, w2), ep
        return None, None
    else:
        # Случайный поиск
        best = None
        for _ in range(trials):
            w0 = tuple(random.uniform(-1, 1) for _ in range(3))
            _, ep = hebb_train_until_converge(w0)
            if ep == target_epochs:
                return w0, ep
            if best is None or ep < best[1]:
                best = (w0, ep)
        return None, None

if __name__ == "__main__":
    # Попробуем сначала сетку для наглядности
    w_init, ep = try_find_initial_weights_for_exact_epochs(target_epochs=3, grid=True)
    if w_init is None:
        # Если на грубой сетке не нашлось, пробуем случайный поиск
        w_init, ep = try_find_initial_weights_for_exact_epochs(target_epochs=3, trials=50000, grid=False)

    if w_init is None:
        print("Не удалось найти начальные веса для ровно 3 эпох за отведённые попытки.")
    else:
        print(f"Нашёл начальные веса (в диапазоне [-1,1]), которые дают ровно 3 эпохи: {w_init}")
        w_final, epochs, history = hebb_train_until_converge(w_init, log=True)
        print(f"Количество эпох до сходимости: {epochs}")
        print(f"Финальные веса: {w_final}")
        # При желании печатаем детальный лог
        for e, epoch_log in enumerate(history, start=1):
            print(f"\nЭпоха {e}:")
            updates = 0
            for (x, y, yhat, w_after, action) in epoch_log:
                mark = "✓" if action == "ok" else "✗"
                if action != "ok":
                    updates += 1
                print(f"  x={x}, y={y}, ŷ={yhat}, действие={action}, w→{w_after} {mark}")
            print(f"  Обновлений в эпохе: {updates}")
