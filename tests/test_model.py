import numpy as np

from neural_heston import domain, heston, model


def test_network_shapes_and_quick_fit():
    x = domain.sample(1024, seed=5)
    price, grad = heston.call_and_grad(x)
    m = model.train({"x": x, "price": price, "grad": grad}, epochs=100, batch=128, log=None)
    p, g = m.predict(x)
    # no positivity check: Black + correction can dip below zero in the far wings (report, section 4)
    assert p.shape == (1024,) and g.shape == (1024, 7)
    assert np.sqrt(np.mean((p - price) ** 2)) < 0.1 * price.std()  # a short run already captures the surface
