from flask import Flask, request, send_file
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

@app.route('/generate-visualization', methods=['POST'])
def generate_visualization():
    data = request.json
    phi = data['phi']
    omega = data['omega']
    k = data['k']
    t_max = data['t']
    x_max = data['x']

    t = np.linspace(0, t_max, 1000)
    x = np.linspace(-x_max, x_max, 1000)

    eta = np.sin(omega * t)
    phi_modulated = phi * (1 + eta / (1 + np.exp(-k * (t - t_max / 2))))
    Psi_cubit = phi_modulated[:, None] * np.sin(omega * t[:, None] * x[None, :])

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(x, t, Psi_cubit, shading='auto', cmap='viridis')
    plt.colorbar(label='Cubit Brane Field Intensity')
    plt.xlabel('Space (x)')
    plt.ylabel('Time (t)')
    plt.title('Cubit Brane Field with Modulated Golden Ratio')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Compute and return the Sigma calculation
@app.route('/compute_sigma', methods=['POST'])
def compute_sigma():
    data = request.json
    p = data['p']
    n = data['n']
    i = data['i']
    e = np.e

    factorial_i = np.math.factorial(i)
    term = (np.log10(p) - factorial_i * n * np.log(10)) / (e * n * np.log(10))
    sigma = 10 ** (np.exp(term))
    return jsonify({"sigma": sigma})

# Generate visualization
@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    t = np.linspace(0, 10, 1000)
    x = np.linspace(-10, 10, 1000)
    omega =
