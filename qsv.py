import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Qiskit imports
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Statevector, partial_trace
from openai import OpenAI

st.set_page_config(page_title="Quantum Bloch Visualizer", layout="wide")

# ---- Flexible Sidebar CSS ----
st.markdown(
    """
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
/* Flexible and modern sidebar width - auto scales up to 480px */
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
  min-width: 280px;
  max-width: 400px;
  width: 100%;
}
</style>
""", unsafe_allow_html=True,
)

def bloch_from_rho(rho):
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z])

def plot_bloch_vector(bloch_vector, qubit_index):
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    sphere = go.Surface(x=xs, y=ys, z=zs, colorscale=[[0, "#F5CBA7"], [1, "#FAD7A0"]], opacity=0.4, showscale=False)
    x_axis = go.Scatter3d(x=[0, 1.1], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="#FF6F61", width=3))
    y_axis = go.Scatter3d(x=[0, 0], y=[0, 1.1], z=[0, 0], mode="lines", line=dict(color="#6FFFE9", width=3))
    z_axis = go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.1], mode="lines", line=dict(color="#FFD166", width=3))
    shaft = go.Scatter3d(
        x=[0, bloch_vector[0] * 0.9],
        y=[0, bloch_vector[1] * 0.9],
        z=[0, bloch_vector[2] * 0.9],
        mode="lines",
        line=dict(color="#00E5FF", width=6),
    )
    arrowhead = go.Cone(
        x=[bloch_vector[0]], y=[bloch_vector[1]], z=[bloch_vector[2]],
        u=[bloch_vector[0] * 0.1], v=[bloch_vector[1] * 0.1], w=[bloch_vector[2] * 0.1],
        showscale=False, colorscale=[[0, "#00E5FF"], [1, "#00E5FF"]],
        sizemode="absolute", sizeref=0.15, anchor="tail",
    )
    fig = go.Figure(data=[sphere, x_axis, y_axis, z_axis, shaft, arrowhead])
    fig.update_layout(
        title=f"<b style='color:white'>Qubit {qubit_index}</b>",
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], autorange=False, showbackground=False, color="white"),
            yaxis=dict(range=[-1.2, 1.2], autorange=False, showbackground=False, color="white"),
            zaxis=dict(range=[-1.2, 1.2], autorange=False, showbackground=False, color="white"),
            aspectmode="cube", bgcolor="#0D1B2A"
        ),
        font=dict(color="white"), paper_bgcolor="#0D1B2A",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

OPENAI_KEY = ""
openai_client = None
if OPENAI_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        openai_client = None

st.title("🔮 Quantum State Visualizer")

# --- Q-Chat in Sidebar ---
st.sidebar.subheader("💬 Q-Chat")
st.sidebar.caption("Ask about your circuit, gates, states, or Bloch spheres.")
user_query = st.sidebar.text_input("Question", key="qchat_input")
if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""
if st.sidebar.button("Ask", key="qchat_ask"):
    if not user_query:
        st.sidebar.warning("Please type a question first.")
    elif not openai_client:
        st.session_state.chat_response = "⚠️ Q-Chat not configured. Put your OPENAI_API_KEY in secrets or env."
    else:
        with st.spinner("Thinking..."):
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a concise and expert assistant on quantum circuits and Bloch spheres. Answer briefly and stay on topic."},
                        {"role": "user", "content": user_query},
                    ],
                    max_tokens=256,
                )
                text = resp.choices[0].message.get("content") if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
                st.session_state.chat_response = text
            except Exception as e:
                st.session_state.chat_response = f"⚠️ Q-Chat error: {e}"
if st.session_state.chat_response:
    st.sidebar.success(st.session_state.chat_response)

# --- Quantum Circuit State ---
if "gates" not in st.session_state:
    st.session_state.gates = []
if "n_qubits" not in st.session_state:
    st.session_state.n_qubits = None
if "evolution_states" not in st.session_state:
    st.session_state.evolution_states = []
if "evolution_valid" not in st.session_state:
    st.session_state.evolution_valid = False
if "evo_step_index" not in st.session_state:
    st.session_state.evo_step_index = 0
if "gates_hash" not in st.session_state:
    st.session_state.gates_hash = None

n_qubits = st.selectbox("Select number of qubits", options=[None, 1, 2, 3, 4, 5, 6], index=0)
if n_qubits and st.session_state.n_qubits != n_qubits:
    st.session_state.n_qubits = n_qubits
    st.session_state.gates = []
    st.session_state.evolution_states = []
    st.session_state.evolution_valid = False
    st.session_state.evo_step_index = 0
    st.session_state.gates_hash = None

if n_qubits:
    st.sidebar.markdown("## Circuit controls")
    st.sidebar.write(f"Circuit currently has {n_qubits} qubit(s).")
    if st.sidebar.button("Reset Circuit", key="reset"):
        st.session_state.gates = []
        st.session_state.evolution_states = []
        st.session_state.evolution_valid = False
        st.session_state.evo_step_index = 0
        st.session_state.gates_hash = None
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.session_state["_force_rerun"] = st.session_state.get("_force_rerun", 0) + 1
            st.stop()

    with st.sidebar.expander("Add a gate", expanded=True):
        gate = st.selectbox("Gate", ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "CX", "SWAP"], key="add_gate")
        target = st.number_input("Target qubit (index)", min_value=0, max_value=n_qubits - 1, value=0, key="add_target")
        theta = None
        if gate in ["Rx", "Ry", "Rz"]:
            theta = st.slider("Angle (radians)", 0.0, 2 * np.pi, 1.5708, key="add_theta")
        ctrl = None
        if gate in ["CX", "SWAP"]:
            ctrl = st.number_input("Second qubit", min_value=0, max_value=n_qubits - 1, value=1, key="add_ctrl")
        if st.button("Add Gate", key="add_gate_button"):
            st.session_state.gates.append((gate, int(target), int(ctrl) if ctrl is not None else None, float(theta) if theta is not None else None))
            st.session_state.evolution_valid = False
            st.session_state.evo_step_index = len(st.session_state.gates)  # jump to newest
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.session_state["_force_rerun"] = st.session_state.get("_force_rerun", 0) + 1
                st.stop()

    st.sidebar.markdown("### Gate Sequence")
    for idx, (g, t, c, th) in enumerate(list(st.session_state.gates)):
        with st.sidebar.expander(f"{g} on q{t}", expanded=False):
            new_gate = st.selectbox("Gate type", ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "CX", "SWAP"], index=["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "CX", "SWAP"].index(g), key=f"gate_type_{idx}")
            new_target = st.number_input("Target qubit", min_value=0, max_value=n_qubits - 1, value=t, key=f"target_{idx}")
            new_theta = th
            if new_gate in ["Rx", "Ry", "Rz"]:
                new_theta = st.slider("Angle (radians)", 0.0, 2 * np.pi, new_theta if new_theta is not None else 1.5708, key=f"theta_{idx}")
            new_ctrl = c
            if new_gate in ["CX", "SWAP"]:
                new_ctrl = st.number_input("Second qubit", min_value=0, max_value=n_qubits - 1, value=c if c is not None else 1, key=f"ctrl_{idx}")
            if st.button("Save", key=f"save_{idx}"):
                st.session_state.gates[idx] = (new_gate, int(new_target), int(new_ctrl) if new_ctrl is not None else None, float(new_theta) if new_theta is not None else None)
                st.session_state.evolution_valid = False
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.session_state["_force_rerun"] = st.session_state.get("_force_rerun", 0) + 1
                    st.stop()
            if st.button("Delete", key=f"delete_{idx}"):
                st.session_state.gates.pop(idx)
                st.session_state.evolution_valid = False
                if st.session_state.evo_step_index >= len(st.session_state.gates):
                    st.session_state.evo_step_index = max(0, len(st.session_state.gates) - 1)
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.session_state["_force_rerun"] = st.session_state.get("_force_rerun", 0) + 1
                    st.stop()

    qc = QuantumCircuit(n_qubits)
    for (g, t, c, th) in st.session_state.gates:
        if g == "H":
            qc.h(t)
        elif g == "X":
            qc.x(t)
        elif g == "Y":
            qc.y(t)
        elif g == "Z":
            qc.z(t)
        elif g == "S":
            qc.s(t)
        elif g == "T":
            qc.t(t)
        elif g == "Rx":
            qc.rx(th, t)
        elif g == "Ry":
            qc.ry(th, t)
        elif g == "Rz":
            qc.rz(th, t)
        elif g == "CX" and c is not None and c != t:
            qc.cx(c, t)
        elif g == "SWAP" and c is not None and c != t:
            qc.swap(c, t)

    st.subheader("Current Circuit")
    st.code(qc.draw(output="text"), language="text")

    try:
        qasm_str = qiskit.qasm2.dumps(qc)
        st.download_button("⬇️ Download QASM", qasm_str, file_name="circuit.qasm")
    except Exception:
        pass

    def compute_evolution_states(gates_list, n_qubits):
        states = []
        base_qc = QuantumCircuit(n_qubits)
        for (g, t, c, th) in gates_list:
            if g == "H":
                base_qc.h(t)
            elif g == "X":
                base_qc.x(t)
            elif g == "Y":
                base_qc.y(t)
            elif g == "Z":
                base_qc.z(t)
            elif g == "S":
                base_qc.s(t)
            elif g == "T":
                base_qc.t(t)
            elif g == "Rx":
                base_qc.rx(th, t)
            elif g == "Ry":
                base_qc.ry(th, t)
            elif g == "Rz":
                base_qc.rz(th, t)
            elif g == "CX" and c is not None and c != t:
                base_qc.cx(c, t)
            elif g == "SWAP" and c is not None and c != t:
                base_qc.swap(c, t)
            try:
                sv = Statevector.from_instruction(base_qc)
                states.append(sv)
            except Exception:
                states.append(None)
        return states

    current_gates_hash = hash(tuple(st.session_state.gates))
    if st.session_state.gates_hash != current_gates_hash or not st.session_state.evolution_valid:
        try:
            st.session_state.evolution_states = compute_evolution_states(st.session_state.gates, n_qubits)
            st.session_state.evolution_valid = True
            st.session_state.gates_hash = current_gates_hash
            if len(st.session_state.evolution_states) > 0:
                st.session_state.evo_step_index = len(st.session_state.evolution_states) - 1
            else:
                st.session_state.evo_step_index = 0
        except Exception as e:
            st.session_state.evolution_states = []
            st.session_state.evolution_valid = False
            st.error(f"Error computing evolution: {e}")

    st.markdown("### State evolution (after each gate)")
    num_steps = len(st.session_state.evolution_states)
    if num_steps == 0:
        st.info("No gates yet — add gates to see state evolution.")
    else:
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("◀ Prev", key="prev_step"):
                st.session_state.evo_step_index = max(0, st.session_state.evo_step_index - 1)
        with col2:
            if num_steps > 1:
                step = st.slider(
                    f"Step (1..{num_steps}) — show state after gate",
                    min_value=1, max_value=num_steps, value=st.session_state.evo_step_index + 1, key="evo_slider"
                )
                st.session_state.evo_step_index = step - 1
            else:
                st.markdown(f"Step 1 of 1 — auto-selected")
                st.session_state.evo_step_index = 0
        with col3:
            if st.button("Next ▶", key="next_step"):
                st.session_state.evo_step_index = min(num_steps - 1, st.session_state.evo_step_index + 1)

        step_idx = st.session_state.evo_step_index
        gate_label = "Initial" if step_idx < 0 else f"After gate {step_idx + 1}: {st.session_state.gates[step_idx][0]} on q{st.session_state.gates[step_idx][1]}"
        st.caption(gate_label)
        sv = st.session_state.evolution_states[step_idx]
        if sv is None:
            st.error("State at this step could not be simulated.")
        else:
            cols = st.columns(min(4, n_qubits))
            for q_idx in range(n_qubits):
                try:
                    other_indices = [i for i in range(n_qubits) if i != q_idx]
                    rho_q = partial_trace(sv, other_indices)
                    rho_np = np.array(rho_q.data, dtype=complex)
                    bloch = bloch_from_rho(rho_np)
                    with cols[q_idx % 4]:
                        st.plotly_chart(plot_bloch_vector(bloch, q_idx + 1), use_container_width=True)
                except Exception as e:
                    with cols[q_idx % 4]:
                        st.error(f"Error for qubit {q_idx}: {e}")

else:
    st.info("👆 Please select the number of qubits to start building your circuit.")
