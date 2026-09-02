from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from router_engine import RoutingPolicy

app = FastAPI(title="Epoxy Routing Policy Demo")
policy = RoutingPolicy(threshold=0.75)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Epoxy Routing Policy Demo</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        .alert { background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; border: 1px solid #ffeeba; margin-bottom: 20px;}
        .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #f9f9f9; }
        input[type="range"] { width: 100%; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; font-family: monospace; background: #2d2d2d; color: #00ff00; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Epoxy Routing Policy Demo</h1>
    <div class="alert">
        <strong>Notice:</strong> This is a lightweight, deterministic public demo of the Epoxy <code>RoutingPolicy</code>. 
        It does <strong>NOT</strong> run the full DistilBERT/LoRA ML inference system, AWS EKS, RabbitMQ, or PostgreSQL backends.
    </div>
    
    <div class="card">
        <h3>Test the Router</h3>
        <p>Adjust the simulated classifier probabilities below. The policy threshold is set to 0.75.</p>
        
        <label for="p_simple">Probability Simple: <span id="val_simple">0.80</span></label>
        <input type="range" id="p_simple" min="0" max="1" step="0.01" value="0.80" oninput="updateProbs(this.value)">
        
        <label for="p_complex">Probability Complex: <span id="val_complex">0.20</span></label>
        <input type="range" id="p_complex" min="0" max="1" step="0.01" value="0.20" disabled>
        
        <button onclick="runRoute()" style="margin-top: 15px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Evaluate Route</button>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        function updateProbs(val) {
            let pSimple = parseFloat(val);
            let pComplex = (1.0 - pSimple).toFixed(2);
            document.getElementById('val_simple').innerText = pSimple.toFixed(2);
            document.getElementById('val_complex').innerText = pComplex;
            document.getElementById('p_complex').value = pComplex;
        }

        async function runRoute() {
            const pSimple = parseFloat(document.getElementById('p_simple').value);
            const pComplex = parseFloat(document.getElementById('p_complex').value);
            
            const response = await fetch('/route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ probabilities: [pSimple, pComplex] })
            });
            
            const data = await response.json();
            const resDiv = document.getElementById('result');
            resDiv.style.display = 'block';
            resDiv.innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""

class RouteRequest(BaseModel):
    probabilities: list[float]

@app.get("/")
def get_home():
    return HTMLResponse(content=html_content)

@app.get("/health")
def get_health():
    return {"status": "ok"}

@app.post("/route")
def calculate_route(request: RouteRequest):
    decision = policy.decide(request.probabilities)
    return decision.to_dict()
