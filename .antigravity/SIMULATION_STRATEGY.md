# 🛠️ Generic Simulation & Offline Development Protocol (v1.0)

## Objective
To accelerate development and allow for "Instant Prototyping" without requiring live API keys, implement a **Simulation Bridge**. This allows the entire stack to run offline or in a mocked state.

## 🎯 Primary Objectives
1.  **Zero-Key Startup**: Allow the development environment to run and be fully interactive even if external service keys are missing.
2.  **Instant Simulation (Default)**: By default, local development uses the **Simulation Bridge**.
3.  **Explicit Connection**: Real integrations are only activated if the user explicitly enables them via environment flags.

---

## 🏛️ Simulation Patterns

### 1. 🔑 Mock Authentication
- **Trigger**: `SIMULATION_MODE=true` (Default).
- **Implementation**:
    - **Middleware/Edge Logic**: **CRITICAL:** The simulation check must happen *before* the authentication wrapper. If in simulation mode, bypass standard auth redirects.
    - **Session Logic**: Return a hardcoded session/token when in simulation mode to allow components to render correctly.
    - **Functional Journey**: The Login/Register pages must simulate a successful flow with a short delay to fulfill the "No-Mockup" requirement.

### 2. 🗄️ Simulated Database
- **Trigger**: `SIMULATE_DB=true`.
- **Implementation**:
    - **Wrapper**: Provide a proxy or wrapper that intercepts database calls.
    - **Storage**: Store data in local JSON files or browser storage for client-only demos.
    - **CRUD**: Implement basic Create, Read, Update, and Delete logic that operates on the local mock data.

### 3. 💳 Third-Party Service Bypassing
- **Trigger**: `DEV_MODE=true`.
- **Implementation**:
    - **Payments**: Intercept payment creation and return a success state immediately without hitting external APIs.
    - **Email/Communication**: Log the content (HTML/Subject) to the developer console instead of attempting to send.

---

## 🛠 Integration Requirements
Architects and Engineers must collaborate to ensure the "Switch" is clean:
1.  **Toggle**: One single flag in the environment configuration should ideally trigger the entire mock stack.
2.  **Telemetry**: When in Simulation Mode, the UI should show a subtle "Simulated Environment" indicator in the corner.

## 🚫 Critical Constraints
- **NEVER** include mock data in the production build.
- **NEVER** bypass real security checks in a production environment.
- Mock data must match the **Aesthetic Archetype** of the product (e.g., use real-sounding names appropriate to the domain).
