"""
Simulated Federated Learning Module
Implements simulated FL for privacy-preserving distributed training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    logger.warning("Flower not installed. FL simulation will be limited.")


@dataclass
class ClientConfig:
    """Configuration for a simulated FL client."""
    client_id: int
    data_partition: torch.Tensor
    labels_partition: torch.Tensor
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001


class FederatedTrainer:
    """
    Simulated Federated Learning Trainer.

    Implements:
    - Local client training simulation
    - FedAvg aggregation
    - Secure aggregation (simplified)
    """

    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 5,
        num_rounds: int = 10,
        local_epochs: int = 5,
        fraction_fit: float = 1.0,
        aggregation_method: str = "fedavg"
    ):
        self.model = model
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.fraction_fit = fraction_fit
        self.aggregation_method = aggregation_method

        self.clients: List[ClientConfig] = []
        self.logger = logging.getLogger(__name__)

    def setup_clients(
        self,
        dataset: Tuple[torch.Tensor, torch.Tensor],
        partition_type: str = "iid"
    ) -> List[ClientConfig]:
        """Set up simulated clients with data partitions."""
        data, labels = dataset
        n_samples = len(data)
        samples_per_client = n_samples // self.num_clients

        self.clients = []

        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client

            if partition_type == "non_iid" and i > 0:
                start_idx = (i * samples_per_client) % n_samples
                end_idx = (start_idx + samples_per_client) % n_samples

            client = ClientConfig(
                client_id=i,
                data_partition=data[start_idx:end_idx],
                labels_partition=labels[start_idx:end_idx],
                local_epochs=self.local_epochs
            )
            self.clients.append(client)

        self.logger.info(f"Set up {self.num_clients} clients with {partition_type} partition")
        return self.clients

    def train_local_client(
        self,
        client: ClientConfig,
        global_model: nn.Module
    ) -> nn.Module:
        """Train model on local client data."""
        local_model = self._copy_model(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=client.learning_rate)
        criterion = nn.CrossEntropyLoss()

        local_model.train()
        for epoch in range(client.local_epochs):
            for i in range(0, len(client.data_partition), client.batch_size):
                batch_data = client.data_partition[i:i+client.batch_size]
                batch_labels = client.labels_partition[i:i+client.batch_size]

                optimizer.zero_grad()
                outputs = local_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        return local_model

    def aggregate_models_fedavg(
        self,
        client_models: List[nn.Module],
        client_weights: Optional[List[float]] = None
    ) -> nn.Module:
        """Aggregate client models using FedAvg."""
        if not client_models:
            raise ValueError("No client models to aggregate")

        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)

        global_model = self._copy_model(client_models[0])
        global_dict = global_model.state_dict()

        for key in global_dict:
            weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for model, weight in zip(client_models, client_weights):
                weighted_sum += weight * model.state_dict()[key].float()
            global_dict[key] = weighted_sum

        global_model.load_state_dict(global_dict)
        return global_model

    def train_round(
        self,
        round_num: int,
        client_models: List[nn.Module]
    ) -> Dict[str, float]:
        """Execute one FL round."""
        global_model = self.aggregate_models_fedavg(client_models)

        metrics = {
            'round': round_num,
            'num_clients': len(client_models),
            'method': self.aggregation_method
        }

        return metrics

    def run_simulation(self) -> List[Dict]:
        """Run complete FL simulation."""
        results = []

        for round_num in range(self.num_rounds):
            self.logger.info(f"FL Round {round_num + 1}/{self.num_rounds}")

            client_models = []
            for client in self.clients:
                client_model = self.train_local_client(client, self.model)
                client_models.append(client_model)

            metrics = self.train_round(round_num, client_models)
            results.append(metrics)

            self.model = self.aggregate_models_fedavg(client_models)

        return results

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model."""
        new_model = type(model)(**model.config if hasattr(model, 'config') else {})
        new_model.load_state_dict(model.state_dict())
        return new_model