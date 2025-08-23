import streamlit as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import plotly.graph_objects as go
import openai

# ---------------- OPENAI SETUP ----------------
openai.api_key = "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"

# ---------------- CSS for Sidebar Buttons ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    color: black !important;
    font-weight: 600 !important;
    background-color: white !important;
    border: 1px solid #ccc !important;
}
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] button:focus,
[data-testid="stSidebar"] button:active,
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button:active {
    color: black !important;
    background-color: white !important;
    border: 1px solid #aaa !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers: partial trace ----------------
def index_from_bits(bits):
    idx = 0
    for b in bits:
        idx = (idx << 1) | (b & 1)
    return idx

def all_other_bit_patterns(n, keep_q):
    m = n - 1
    for t in range(1 << m):
        yield [(t >> (m - 1 - j)) & 1 for j in range(m)]

def reduced_density_from_statevector(statevector, keep_q, n):
    psi = np.asarray(statevector).reshape(-1)
    rho_full = np.outer(psi, np.conj(psi))
    rho_keep = np.zeros((2, 2), dtype=complex)

    for i in (0, 1):
        for k in (0, 1):
            s = 0 + 0j
            for t_bits in all_other_bit_patterns(n, keep_q):
                full_row = t_bits.copy()
                full_col = t_bits.copy()
                full_row.insert(keep_q, i)
                full_col.insert(keep_q, k)
                row_idx = index_from_bits(full_row)
                col_idx = index_from_bits(full_col)
                s += rho_full[row_idx, col_idx]
            rho_keep[i, k] = s
    return rho_keep

def bloch_from_rho(rho):
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z])

# ---------------- Plot ----------------
def plot_bloch_vector(bloch_vector, qubit_index):
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    sphere = go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, "#F5CBA7"], [1, "#FAD7A0"]],
        opacity=0.4,
        showscale=False
    )

    x_axis = go.Scatter3d(x=[0, 1.1], y=[0, 0], z=[0, 0],
                          mode="lines", line=dict(color="#FF6F61", width=3))
    y_axis = go.Scatter3d(x=[0, 0], y=[0, 1.1], z=[0, 0],
                          mode="lines", line=dict(color="#6FFFE9", width=3))
    z_axis = go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.1],
                          mode="lines", line=dict(color="#FFD166", width=3))

    shaft = go.Scatter3d(
        x=[0, bloch_vector[0] * 0.9],
        y=[0, bloch_vector[1] * 0.9],
        z=[0, bloch_vector[2] * 0.9],
        mode="lines",
        line=dict(color="#00E5FF", width=6)
    )

    arrowhead = go.Cone(
        x=[bloch_vector[0]],
        y=[bloch_vector[1]],
        z=[bloch_vector[2]],
        u=[bloch_vector[0] * 0.1],
        v=[bloch_vector[1] * 0.1],
        w=[bloch_vector[2] * 0.1],
        showscale=False,
        colorscale=[[0, "#00E5FF"], [1, "#00E5FF"]],
        sizemode="absolute",
        sizeref=0.15,
        anchor="tail"
    )

    fig = go.Figure(data=[sphere, x_axis, y_axis, z_axis, shaft, arrowhead])
    fig.update_layout(
        title=f"<b style='color:white'>Qubit {qubit_index}</b>",
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], autorange=False, showbackground=False, color="white"),
            yaxis=dict(range=[-1.2, 1.2], autorange=False, showbackground=False, color="white"),
            zaxis=dict(range=[-1.2, 1.2], autorange=False, showbackground=False, color="white"),
            aspectmode="cube",
            bgcolor="#0D1B2A"
        ),
        font=dict(color="white"),
        paper_bgcolor="#0D1B2A",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Quantum Bloch Visualizer", layout="wide")
st.title("🔮 Quantum State Visualizer")

# ---------------- Chatbot ----------------
st.sidebar.subheader("💬 Q-Chat")
user_query = st.sidebar.text_input("Ask about your circuit, gates, states, or Bloch spheres:")

# Store chatbot response in session_state for proper placement
if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""

# Trigger chatbot ONLY when Ask button is clicked
if st.sidebar.button("Ask") and user_query:
    with st.spinner("Thinking..."):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise and expert assistant on quantum circuits, gates, and Bloch spheres. Always answer briefly and only about the user's current circuit."
                },
                {"role": "user", "content": user_query}
            ]
        )
    st.session_state.chat_response = response.choices[0].message.content

# Show chatbot response immediately below input
if st.session_state.chat_response:
    st.sidebar.success(st.session_state.chat_response)

# ---------------- Quantum Circuit UI ----------------
n_qubits = st.selectbox("Select number of qubits", options=[None, 1, 2, 3, 4, 5, 6], index=0)

if n_qubits:
    if "qc" not in st.session_state:
        st.session_state.qc = QuantumCircuit(n_qubits)
        st.session_state.n_qubits = n_qubits
        st.session_state.gates = []
    elif st.session_state.n_qubits != n_qubits:
        st.session_state.qc = QuantumCircuit(n_qubits)
        st.session_state.n_qubits = n_qubits
        st.session_state.gates = []

    st.sidebar.markdown("## Circuit controls")
    st.sidebar.write(f"Circuit currently has {n_qubits} qubits.")

    if st.sidebar.button("Reset Circuit"):
        st.session_state.qc = QuantumCircuit(n_qubits)
        st.session_state.gates = []
        st.success("Circuit reset.")

    with st.sidebar.expander("Add a gate", expanded=True):
        gate = st.selectbox("Gate", ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "CX", "SWAP"])
        target = st.number_input("Target qubit (index)", min_value=0, max_value=n_qubits-1, value=0)
        theta = None
        if gate in ["Rx", "Ry", "Rz"]:
            theta = st.slider("Angle (radians)", 0.0, 6.283185, 1.5708)
        ctrl = None
        if gate in ["CX", "SWAP"]:
            ctrl = st.number_input("Second qubit", min_value=0, max_value=n_qubits-1, value=1)

        if st.button("Add Gate"):
            st.session_state.gates.append((gate, target, ctrl, theta))

    # ---------------- Gate Sequence Section ----------------
    st.sidebar.markdown("### Gate Sequence")
    for idx in range(len(st.session_state.gates)):
        gate, target, ctrl, theta = st.session_state.gates[idx]
        with st.sidebar.expander(f"{gate} on q{target}", expanded=False):
            new_gate = st.selectbox(
                "Gate type",
                ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "CX", "SWAP"],
                index=["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "CX", "SWAP"].index(gate),
                key=f"gate_type_{idx}"
            )
            new_target = st.number_input(
                "Target qubit", min_value=0, max_value=n_qubits - 1, value=target, key=f"target_{idx}"
            )
            new_theta = theta
            if new_gate in ["Rx", "Ry", "Rz"]:
                new_theta = st.slider(
                    "Angle (radians)", 0.0, 6.283185, new_theta if new_theta else 1.5708, key=f"theta_{idx}"
                )
            new_ctrl = ctrl
            if new_gate in ["CX", "SWAP"]:
                new_ctrl = st.number_input(
                    "Second qubit", min_value=0, max_value=n_qubits - 1,
                    value=ctrl if ctrl is not None else 1,
                    key=f"ctrl_{idx}"
                )

            if st.button("Save", key=f"save_{idx}"):
                st.session_state.gates[idx] = (new_gate, new_target, new_ctrl, new_theta)
                st.rerun()  # FIXED

            if st.button("Delete", key=f"delete_{idx}"):
                st.session_state.gates.pop(idx)
                st.rerun()  # FIXED

    # Build the circuit
    qc = QuantumCircuit(n_qubits)
    for gate, target, ctrl, theta in st.session_state.gates:
        if gate == "H": qc.h(target)
        elif gate == "X": qc.x(target)
        elif gate == "Y": qc.y(target)
        elif gate == "Z": qc.z(target)
        elif gate == "S": qc.s(target)
        elif gate == "T": qc.t(target)
        elif gate == "Rx": qc.rx(theta, target)
        elif gate == "Ry": qc.ry(theta, target)
        elif gate == "Rz": qc.rz(theta, target)
        elif gate == "CX" and ctrl is not None and ctrl != target:
            qc.cx(ctrl, target)
        elif gate == "SWAP" and ctrl is not None and ctrl != target:
            qc.swap(ctrl, target)
    st.session_state.qc = qc

    st.subheader("Current Circuit (text view)")
    st.code(qc.draw(output="text"), language="text")

    # Visualize Bloch spheres
    if st.button("Visualize Bloch Spheres"):
        try:
            sv = Statevector.from_instruction(qc)
        except Exception as e:
            st.error(f"Error simulating circuit: {e}")
            st.stop()

        cols = st.columns(min(4, n_qubits))
        for display_q in range(n_qubits):
            state_q_index = n_qubits - 1 - display_q
            rho_q = reduced_density_from_statevector(sv.data, state_q_index, n_qubits)
            bloch = bloch_from_rho(rho_q)
            with cols[display_q % 4]:
                st.plotly_chart(plot_bloch_vector(bloch, display_q + 1), use_container_width=True)
else:
    st.info("👆 Please select the number of qubits to start building your circuit.")
