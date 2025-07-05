"""
Advanced Optimization and Simulation Engine
Comprehensive optimization algorithms and environmental simulations
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d, griddata
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import sympy as sp
from sympy import symbols, diff, integrate, solve, lambdify
import warnings
warnings.filterwarnings('ignore')

class RiverEcosystemSimulator:
    """
    Advanced river ecosystem simulation using differential equations
    """
    
    def __init__(self):
        self.parameters = {
            'carrying_capacity': 1000,  # Maximum fish population
            'growth_rate': 0.1,  # Fish population growth rate
            'pollution_decay': 0.05,  # Pollution natural decay rate
            'oxygen_production': 0.8,  # Oxygen production by plants
            'oxygen_consumption': 0.3,  # Oxygen consumption by fish
            'temperature_effect': 0.02,  # Temperature effect on growth
            'ph_optimum': 7.0,  # Optimal pH for ecosystem
            'flow_rate_base': 50.0,  # Base river flow rate
        }
        
        self.state_variables = [
            'fish_population',
            'pollution_level',
            'dissolved_oxygen',
            'plant_biomass',
            'temperature',
            'ph_level',
            'nutrient_level'
        ]
        
        self.simulation_results = None
        
    def ecosystem_dynamics(self, t, y, external_inputs=None):
        """
        Define the differential equations for ecosystem dynamics
        """
        fish_pop, pollution, oxygen, plants, temp, ph, nutrients = y
        
        # External inputs (human activities, weather, etc.)
        if external_inputs is None:
            pollution_input = 0.1 * np.sin(0.1 * t)  # Periodic pollution
            temperature_forcing = 20 + 5 * np.sin(2 * np.pi * t / 365)  # Seasonal temperature
            nutrient_input = 0.05
        else:
            pollution_input = external_inputs.get('pollution_input', 0)
            temperature_forcing = external_inputs.get('temperature', temp)
            nutrient_input = external_inputs.get('nutrient_input', 0)
        
        # Fish population dynamics (logistic growth with environmental factors)
        ph_stress = np.exp(-0.5 * ((ph - self.parameters['ph_optimum']) / 0.5) ** 2)
        oxygen_stress = np.tanh(oxygen / 5.0)  # Oxygen stress function
        pollution_stress = np.exp(-pollution / 10.0)  # Pollution stress
        
        fish_growth = (self.parameters['growth_rate'] * fish_pop * 
                      (1 - fish_pop / self.parameters['carrying_capacity']) *
                      ph_stress * oxygen_stress * pollution_stress)
        
        # Pollution dynamics
        pollution_change = (pollution_input - 
                           self.parameters['pollution_decay'] * pollution -
                           0.01 * plants * pollution)  # Plants help clean pollution
        
        # Dissolved oxygen dynamics
        oxygen_production = self.parameters['oxygen_production'] * plants * (1 - oxygen / 15.0)
        oxygen_consumption = (self.parameters['oxygen_consumption'] * fish_pop + 
                             0.02 * pollution)  # Pollution consumes oxygen
        oxygen_change = oxygen_production - oxygen_consumption
        
        # Plant biomass dynamics
        light_availability = 1.0 / (1 + 0.1 * pollution)  # Pollution reduces light
        nutrient_limitation = nutrients / (nutrients + 1.0)  # Michaelis-Menten kinetics
        plant_growth = 0.2 * plants * light_availability * nutrient_limitation * (1 - plants / 100)
        plant_decay = 0.05 * plants
        plants_change = plant_growth - plant_decay
        
        # Temperature dynamics (simplified thermal model)
        temp_change = 0.1 * (temperature_forcing - temp)
        
        # pH dynamics (influenced by biological processes)
        ph_change = 0.01 * (7.0 - ph) + 0.005 * (plants - 50) - 0.002 * pollution
        
        # Nutrient dynamics
        nutrient_change = (nutrient_input - 0.1 * nutrients - 
                          0.02 * plants * nutrients / (nutrients + 0.5))
        
        return [fish_growth, pollution_change, oxygen_change, plants_change,
                temp_change, ph_change, nutrient_change]
    
    def run_simulation(self, initial_conditions, time_span, external_inputs=None):
        """
        Run the ecosystem simulation
        """
        t_eval = np.linspace(time_span[0], time_span[1], 1000)
        
        # Solve the system of ODEs
        solution = solve_ivp(
            fun=lambda t, y: self.ecosystem_dynamics(t, y, external_inputs),
            t_span=time_span,
            y0=initial_conditions,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6
        )
        
        # Store results
        self.simulation_results = pd.DataFrame(
            solution.y.T,
            columns=self.state_variables,
            index=solution.t
        )
        
        return self.simulation_results
    
    def analyze_stability(self, equilibrium_point):
        """
        Analyze the stability of an equilibrium point using Jacobian eigenvalues
        """
        # Create symbolic variables
        y_sym = symbols(' '.join(self.state_variables))
        t_sym = symbols('t')
        
        # Define symbolic equations (simplified version)
        fish_pop, pollution, oxygen, plants, temp, ph, nutrients = y_sym
        
        # Simplified symbolic dynamics
        eqs = [
            0.1 * fish_pop * (1 - fish_pop / 1000) - 0.01 * pollution * fish_pop,
            0.1 - 0.05 * pollution,
            0.8 * plants - 0.3 * fish_pop,
            0.2 * plants * (1 - plants / 100),
            0.1 * (20 - temp),
            0.01 * (7 - ph),
            0.05 - 0.1 * nutrients
        ]
        
        # Calculate Jacobian matrix
        jacobian = sp.Matrix([[diff(eq, var) for var in y_sym] for eq in eqs])
        
        # Evaluate at equilibrium point
        jacobian_numeric = np.array(jacobian.subs(dict(zip(y_sym, equilibrium_point)))).astype(float)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(jacobian_numeric)
        
        # Determine stability
        is_stable = all(np.real(eigenvalues) < 0)
        
        return {
            'eigenvalues': eigenvalues,
            'is_stable': is_stable,
            'jacobian': jacobian_numeric
        }

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for environmental management
    """
    
    def __init__(self):
        self.optimization_history = []
        self.pareto_front = None
        
    def river_management_objectives(self, x):
        """
        Define multiple objectives for river management
        x = [treatment_level, flow_regulation, monitoring_frequency, conservation_effort]
        """
        treatment_level, flow_regulation, monitoring_freq, conservation = x
        
        # Objective 1: Minimize environmental impact (negative = better)
        pollution_reduction = treatment_level * 0.8
        flow_stability = 1 - abs(flow_regulation - 0.5) * 2
        biodiversity_impact = conservation * 0.9
        env_impact = -(pollution_reduction + flow_stability + biodiversity_impact) / 3
        
        # Objective 2: Minimize economic cost
        treatment_cost = treatment_level ** 2 * 1000000  # Non-linear cost
        infrastructure_cost = flow_regulation * 500000
        monitoring_cost = monitoring_freq * 100000
        conservation_cost = conservation * 300000
        total_cost = treatment_cost + infrastructure_cost + monitoring_cost + conservation_cost
        
        # Objective 3: Maximize social benefit (negative = maximize)
        water_quality_benefit = treatment_level * 0.9
        flood_protection = flow_regulation * 0.7
        recreation_value = conservation * 0.6
        health_benefit = (treatment_level + monitoring_freq) * 0.4
        social_benefit = -(water_quality_benefit + flood_protection + recreation_value + health_benefit)
        
        return [env_impact, total_cost, social_benefit]
    
    def nsga2_selection(self, population, objectives, population_size):
        """
        Non-dominated Sorting Genetic Algorithm II selection
        """
        # Calculate domination relationships
        domination_count = np.zeros(len(population))
        dominated_solutions = [[] for _ in range(len(population))]
        
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
        
        # Non-dominated sorting
        fronts = []
        current_front = []
        
        for i in range(len(population)):
            if domination_count[i] == 0:
                current_front.append(i)
        
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        # Select solutions
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= population_size:
                selected_indices.extend(front)
            else:
                # Calculate crowding distance and select best
                remaining_slots = population_size - len(selected_indices)
                crowding_distances = self.calculate_crowding_distance(
                    [objectives[i] for i in front]
                )
                sorted_front = sorted(zip(front, crowding_distances), 
                                    key=lambda x: x[1], reverse=True)
                selected_indices.extend([idx for idx, _ in sorted_front[:remaining_slots]])
                break
        
        return selected_indices
    
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    def calculate_crowding_distance(self, objectives):
        """Calculate crowding distance for diversity preservation"""
        if len(objectives) <= 2:
            return [float('inf')] * len(objectives)
        
        distances = [0.0] * len(objectives)
        n_obj = len(objectives[0])
        
        for m in range(n_obj):
            # Sort by objective m
            sorted_indices = sorted(range(len(objectives)), key=lambda i: objectives[i][m])
            
            # Set boundary points to infinity
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate distances for intermediate points
            obj_range = objectives[sorted_indices[-1]][m] - objectives[sorted_indices[0]][m]
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1]][m] - 
                        objectives[sorted_indices[i - 1]][m]
                    ) / obj_range
        
        return distances
    
    def optimize_river_management(self, generations=100, population_size=50):
        """
        Optimize river management using multi-objective genetic algorithm
        """
        # Initialize population
        population = np.random.uniform(0, 1, (population_size, 4))
        
        for generation in range(generations):
            # Evaluate objectives
            objectives = [self.river_management_objectives(individual) for individual in population]
            
            # Selection
            selected_indices = self.nsga2_selection(population, objectives, population_size // 2)
            selected_population = population[selected_indices]
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % len(selected_population)]
                
                # Crossover
                alpha = np.random.random()
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2
                
                # Mutation
                mutation_rate = 0.1
                if np.random.random() < mutation_rate:
                    child1 += np.random.normal(0, 0.1, len(child1))
                if np.random.random() < mutation_rate:
                    child2 += np.random.normal(0, 0.1, len(child2))
                
                # Ensure bounds
                child1 = np.clip(child1, 0, 1)
                child2 = np.clip(child2, 0, 1)
                
                new_population.extend([child1, child2])
            
            # Combine populations
            population = np.vstack([selected_population, new_population])
            
            # Store progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Population size {len(population)}")
        
        # Final evaluation
        final_objectives = [self.river_management_objectives(individual) for individual in population]
        
        # Extract Pareto front
        pareto_indices = []
        for i in range(len(population)):
            is_dominated = False
            for j in range(len(population)):
                if i != j and self.dominates(final_objectives[j], final_objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        
        self.pareto_front = {
            'solutions': population[pareto_indices],
            'objectives': [final_objectives[i] for i in pareto_indices]
        }
        
        return self.pareto_front

class BayesianOptimizer:
    """
    Bayesian optimization for efficient parameter tuning
    """
    
    def __init__(self, bounds, acquisition_function='ei'):
        self.bounds = bounds
        self.acquisition_function = acquisition_function
        self.X_observed = []
        self.y_observed = []
        self.gp = None
        
    def expected_improvement(self, X, xi=0.01):
        """
        Expected Improvement acquisition function
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        
        # Current best observation
        f_best = np.max(self.y_observed)
        
        # Calculate expected improvement
        with np.errstate(divide='warn'):
            imp = mu - f_best - xi
            Z = imp / sigma.reshape(-1, 1)
            ei = imp * norm.cdf(Z) + sigma.reshape(-1, 1) * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.flatten()
    
    def upper_confidence_bound(self, X, kappa=2.576):
        """
        Upper Confidence Bound acquisition function
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def optimize_water_treatment_system(self, n_iterations=50):
        """
        Optimize water treatment system parameters using Bayesian optimization
        """
        def objective_function(params):
            """
            Objective function for water treatment optimization
            params: [chemical_dosage, retention_time, pH_adjustment, filtration_rate]
            """
            chemical_dosage, retention_time, ph_adjustment, filtration_rate = params
            
            # Simulate treatment efficiency
            base_efficiency = 0.8
            
            # Chemical dosage effect (optimal around 0.6)
            chemical_effect = 1 - 2 * (chemical_dosage - 0.6) ** 2
            
            # Retention time effect (logarithmic improvement)
            retention_effect = np.log(1 + retention_time * 10) / np.log(11)
            
            # pH adjustment effect (optimal around 7.0 mapped to 0.5)
            ph_target = 7.0
            current_ph = 6.0 + 2 * ph_adjustment  # Map [0,1] to [6,8]
            ph_effect = np.exp(-0.5 * ((current_ph - ph_target) / 0.5) ** 2)
            
            # Filtration rate effect
            filtration_effect = filtration_rate * (1 - 0.3 * filtration_rate)  # Optimal around 0.8
            
            # Combine effects
            efficiency = (base_efficiency * chemical_effect * retention_effect * 
                         ph_effect * filtration_effect)
            
            # Add some noise
            efficiency += np.random.normal(0, 0.02)
            
            return efficiency
        
        # Initialize with random samples
        n_initial = 5
        X_init = np.random.uniform(
            low=[bound[0] for bound in self.bounds],
            high=[bound[1] for bound in self.bounds],
            size=(n_initial, len(self.bounds))
        )
        
        y_init = [objective_function(x) for x in X_init]
        
        self.X_observed = X_init.tolist()
        self.y_observed = y_init
        
        # Bayesian optimization loop
        for iteration in range(n_iterations):
            # Fit Gaussian Process
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            
            self.gp.fit(self.X_observed, self.y_observed)
            
            # Optimize acquisition function
            def neg_acquisition(x):
                x = x.reshape(1, -1)
                if self.acquisition_function == 'ei':
                    return -self.expected_improvement(x)[0]
                elif self.acquisition_function == 'ucb':
                    return -self.upper_confidence_bound(x)[0]
            
            # Multiple random starts for global optimization
            best_x = None
            best_acq = float('inf')
            
            for _ in range(10):
                x0 = np.random.uniform(
                    low=[bound[0] for bound in self.bounds],
                    high=[bound[1] for bound in self.bounds]
                )
                
                result = minimize(
                    neg_acquisition,
                    x0,
                    bounds=self.bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            
            # Evaluate objective at new point
            y_new = objective_function(best_x)
            
            # Update observations
            self.X_observed.append(best_x)
            self.y_observed.append(y_new)
            
            if iteration % 10 == 0:
                current_best = np.max(self.y_observed)
                print(f"Iteration {iteration}: Best efficiency = {current_best:.4f}")
        
        # Return best solution
        best_idx = np.argmax(self.y_observed)
        best_solution = {
            'parameters': self.X_observed[best_idx],
            'efficiency': self.y_observed[best_idx],
            'parameter_names': ['chemical_dosage', 'retention_time', 'pH_adjustment', 'filtration_rate']
        }
        
        return best_solution

class NetworkAnalyzer:
    """
    Network analysis for river system connectivity and contamination spread
    """
    
    def __init__(self):
        self.river_network = nx.DiGraph()
        self.contamination_model = None
        
    def create_river_network(self, n_nodes=20, connectivity=0.3):
        """
        Create a realistic river network topology
        """
        # Generate random positions for nodes
        positions = np.random.uniform(0, 100, (n_nodes, 2))
        
        # Add nodes to network
        for i in range(n_nodes):
            self.river_network.add_node(i, pos=positions[i], 
                                      flow_rate=np.random.uniform(10, 100),
                                      elevation=np.random.uniform(0, 50))
        
        # Add edges based on elevation and distance
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Calculate distance
                    dist = np.linalg.norm(positions[i] - positions[j])
                    
                    # Add edge if downstream (lower elevation) and within reasonable distance
                    if (self.river_network.nodes[j]['elevation'] < 
                        self.river_network.nodes[i]['elevation'] and
                        dist < 30 and np.random.random() < connectivity):
                        
                        self.river_network.add_edge(i, j, 
                                                  distance=dist,
                                                  flow_time=dist / 10)  # Travel time
        
        return self.river_network
    
    def simulate_contamination_spread(self, source_node, initial_concentration=100, 
                                    decay_rate=0.1, simulation_time=100):
        """
        Simulate contamination spread through river network
        """
        # Initialize concentrations
        concentrations = {node: 0.0 for node in self.river_network.nodes()}
        concentrations[source_node] = initial_concentration
        
        time_series = []
        
        for t in range(simulation_time):
            new_concentrations = concentrations.copy()
            
            # For each node, calculate contamination flow
            for node in self.river_network.nodes():
                # Natural decay
                new_concentrations[node] *= (1 - decay_rate)
                
                # Flow from upstream nodes
                for upstream in self.river_network.predecessors(node):
                    edge_data = self.river_network.edges[upstream, node]
                    flow_fraction = 0.1  # Fraction of contamination that flows downstream
                    
                    # Add contamination from upstream
                    new_concentrations[node] += (concentrations[upstream] * 
                                                flow_fraction)
            
            concentrations = new_concentrations
            time_series.append(concentrations.copy())
        
        return time_series
    
    def analyze_network_properties(self):
        """
        Analyze structural properties of the river network
        """
        properties = {
            'num_nodes': self.river_network.number_of_nodes(),
            'num_edges': self.river_network.number_of_edges(),
            'density': nx.density(self.river_network),
            'is_connected': nx.is_weakly_connected(self.river_network),
            'num_components': nx.number_weakly_connected_components(self.river_network)
        }
        
        # Calculate centrality measures
        try:
            properties['betweenness_centrality'] = nx.betweenness_centrality(self.river_network)
            properties['closeness_centrality'] = nx.closeness_centrality(self.river_network)
            properties['degree_centrality'] = nx.degree_centrality(self.river_network)
        except:
            properties['centrality_measures'] = "Could not calculate due to network structure"
        
        # Find critical nodes (high centrality)
        if 'betweenness_centrality' in properties:
            sorted_betweenness = sorted(properties['betweenness_centrality'].items(), 
                                      key=lambda x: x[1], reverse=True)
            properties['most_critical_nodes'] = sorted_betweenness[:3]
        
        return properties
    
    def optimize_monitoring_stations(self, num_stations=5):
        """
        Optimize placement of monitoring stations using centrality measures
        """
        if self.river_network.number_of_nodes() == 0:
            return []
        
        # Calculate multiple centrality measures
        betweenness = nx.betweenness_centrality(self.river_network)
        closeness = nx.closeness_centrality(self.river_network)
        
        # Combine centrality measures
        combined_centrality = {}
        for node in self.river_network.nodes():
            combined_centrality[node] = (0.6 * betweenness.get(node, 0) + 
                                       0.4 * closeness.get(node, 0))
        
        # Select top nodes
        sorted_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        optimal_stations = [node for node, centrality in sorted_nodes[:num_stations]]
        
        return {
            'optimal_stations': optimal_stations,
            'centrality_scores': {node: combined_centrality[node] for node in optimal_stations}
        }

def run_comprehensive_optimization_demo():
    """
    Demonstrate the comprehensive optimization and simulation suite
    """
    print("🚀 Starting Comprehensive Optimization and Simulation Demo...")
    
    # 1. River Ecosystem Simulation
    print("\n🌊 Running River Ecosystem Simulation...")
    simulator = RiverEcosystemSimulator()
    
    initial_conditions = [500, 2.0, 8.0, 50, 20, 7.0, 3.0]  # Initial ecosystem state
    time_span = [0, 365]  # One year simulation
    
    results = simulator.run_simulation(initial_conditions, time_span)
    print(f"✅ Ecosystem simulation completed: {len(results)} time points")
    
    # Analyze stability
    equilibrium = [600, 1.5, 9.0, 60, 18, 7.2, 2.5]
    stability = simulator.analyze_stability(equilibrium)
    print(f"🔍 Stability analysis: {'Stable' if stability['is_stable'] else 'Unstable'} equilibrium")
    
    # 2. Multi-objective Optimization
    print("\n🎯 Running Multi-objective Optimization...")
    mo_optimizer = MultiObjectiveOptimizer()
    pareto_front = mo_optimizer.optimize_river_management(generations=50, population_size=30)
    print(f"✅ Multi-objective optimization completed: {len(pareto_front['solutions'])} Pareto solutions")
    
    # 3. Bayesian Optimization
    print("\n🧠 Running Bayesian Optimization...")
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  # Parameter bounds
    bay_optimizer = BayesianOptimizer(bounds, acquisition_function='ei')
    best_treatment = bay_optimizer.optimize_water_treatment_system(n_iterations=30)
    print(f"✅ Bayesian optimization completed: Best efficiency = {best_treatment['efficiency']:.4f}")
    
    # 4. Network Analysis
    print("\n🕸️  Running Network Analysis...")
    network_analyzer = NetworkAnalyzer()
    river_network = network_analyzer.create_river_network(n_nodes=15, connectivity=0.4)
    
    # Analyze network properties
    network_props = network_analyzer.analyze_network_properties()
    print(f"📊 Network analysis: {network_props['num_nodes']} nodes, {network_props['num_edges']} edges")
    
    # Simulate contamination spread
    contamination_spread = network_analyzer.simulate_contamination_spread(
        source_node=0, simulation_time=50
    )
    print(f"🦠 Contamination simulation: {len(contamination_spread)} time steps")
    
    # Optimize monitoring stations
    optimal_stations = network_analyzer.optimize_monitoring_stations(num_stations=3)
    print(f"📍 Optimal monitoring stations: {optimal_stations['optimal_stations']}")
    
    print("\n🎉 Comprehensive Optimization and Simulation Demo Completed!")
    print(f"📋 Demo included:")
    print(f"   - River ecosystem dynamics simulation")
    print(f"   - Stability analysis using Jacobian eigenvalues")
    print(f"   - Multi-objective optimization with NSGA-II")
    print(f"   - Bayesian optimization for parameter tuning")
    print(f"   - River network analysis and contamination modeling")
    print(f"   - Optimal monitoring station placement")
    
    return {
        'ecosystem_results': results,
        'stability_analysis': stability,
        'pareto_front': pareto_front,
        'best_treatment': best_treatment,
        'network_properties': network_props,
        'contamination_spread': contamination_spread,
        'optimal_stations': optimal_stations
    }

if __name__ == "__main__":
    # Run the comprehensive demonstration
    results = run_comprehensive_optimization_demo()
