import pytorch_lightning as pl
import torch
import torchmetrics
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief
from torch import Tensor, nn


class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # date
        date_h_dim = 100
        date_dim = 100
        self.date_input = nn.Linear(1, date_h_dim)
        self.date_h = nn.Linear(date_h_dim, date_dim)
        # coordinate
        coordinate_h_dim = 100
        coordinate_dim = 100
        self.coordinate_input = nn.Linear(2, coordinate_h_dim)
        self.coordinate_h = nn.Linear(coordinate_h_dim, coordinate_dim)
        # co
        co_h_dim = 100
        co_dim = 100
        self.co_input = nn.Linear(5, co_h_dim)
        self.co_h = nn.Linear(co_h_dim, co_dim)
        # o3
        o3_h_dim = 100
        o3_dim = 100
        self.o3_input = nn.Linear(5, o3_h_dim)
        self.o3_h = nn.Linear(o3_h_dim, o3_dim)
        # so2
        so2_h_dim = 100
        so2_dim = 100
        self.so2_input = nn.Linear(5, so2_h_dim)
        self.so2_h = nn.Linear(so2_h_dim, so2_dim)
        # no2
        no2_h_dim = 100
        no2_dim = 100
        self.no2_input = nn.Linear(5, no2_h_dim)
        self.no2_h = nn.Linear(no2_h_dim, no2_dim)
        # temperature
        temperature_h_dim = 100
        temperature_dim = 100
        self.temperature_input = nn.Linear(5, temperature_h_dim)
        self.temperature_h = nn.Linear(temperature_h_dim, temperature_dim)
        # humidity
        humidity_h_dim = 100
        humidity_dim = 100
        self.humidity_input = nn.Linear(5, humidity_h_dim)
        self.humidity_h = nn.Linear(humidity_h_dim, humidity_dim)
        # pressure
        pressure_h_dim = 100
        pressure_dim = 100
        self.pressure_input = nn.Linear(5, pressure_h_dim)
        self.pressure_h = nn.Linear(pressure_h_dim, pressure_dim)
        # ws
        ws_h_dim = 100
        ws_dim = 100
        self.ws_input = nn.Linear(5, ws_h_dim)
        self.ws_h = nn.Linear(ws_h_dim, ws_dim)
        # dew
        dew_h_dim = 100
        dew_dim = 100
        self.dew_input = nn.Linear(5, dew_h_dim)
        self.dew_h = nn.Linear(dew_h_dim, dew_dim)

        feature_dim = (
            date_dim
            + coordinate_dim
            + co_dim
            + o3_dim
            + so2_dim
            + no2_dim
            + temperature_dim
            + humidity_dim
            + pressure_dim
            + ws_dim
            + dew_dim
        )
        self.linear1 = nn.Linear(feature_dim, feature_dim * 3)
        self.linear2 = nn.Linear(feature_dim * 3, int(feature_dim / 2))
        self.linear3 = nn.Linear(int(feature_dim / 2), 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.criterion = self.create_criterion()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, date, coordinate, co, o3, so2, no2, temperature, humidity, pressure, ws, dew):
        x_date = self.date_input(date)
        x_date = self.date_h(x_date)
        x_coordinate = self.coordinate_input(coordinate)
        x_coordinate = self.coordinate_h(x_coordinate)
        x_co = self.co_input(co)
        x_co = self.co_h(x_co)
        x_o3 = self.o3_input(o3)
        x_o3 = self.o3_h(x_o3)
        x_so2 = self.so2_input(so2)
        x_so2 = self.so2_h(x_so2)
        x_no2 = self.no2_input(no2)
        x_no2 = self.no2_h(x_no2)
        x_temperature = self.temperature_input(temperature)
        x_temperature = self.temperature_h(x_temperature)
        x_humidity = self.humidity_input(humidity)
        x_humidity = self.humidity_h(x_humidity)
        x_pressure = self.pressure_input(pressure)
        x_pressure = self.pressure_h(x_pressure)
        x_ws = self.ws_input(ws)
        x_ws = self.ws_h(x_ws)
        x_dew = self.dew_input(dew)
        x_dew = self.dew_h(x_dew)

        x = torch.cat(
            [
                x_date,
                x_coordinate,
                x_co,
                x_o3,
                x_so2,
                x_no2,
                x_temperature,
                x_humidity,
                x_pressure,
                x_ws,
                x_dew,
            ],
            dim=1,
        )
        # batch_size, vec_size = x.size()

        x = self.relu(self.linear1(x))
        x = self.drop(x)
        x = self.relu(self.linear2(x))
        x = self.drop(x)
        x: Tensor = self.linear3(x)
        return x.flatten().float()

    def training_step(self, batch: dict, batch_idx):
        labels: Tensor = batch.pop("label")

        y_preds: Tensor = self.forward(**batch)
        y_preds = y_preds.float()
        loss: Tensor = self.criterion(y_preds.float(), labels.float())
        loss = loss.float()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        labels: Tensor = batch.pop("label")
        y_preds: Tensor = self.forward(**batch)
        y_preds = y_preds.float()
        labels = labels.float()
        loss: Tensor = self.criterion(y_preds, labels)
        loss = loss.float()
        self.rmse(y_preds, labels)

        self.log("val_loss", loss)
        self.log("RMSE", self.rmse)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=2e-5)
        # optimizer = RangerAdaBelief(self.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))
        # optimizer = AdaBelief(
        #     self.parameters(), lr=1e-3, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False
        # )
        return optimizer

    def create_criterion(self):
        # criterion = nn.HuberLoss()
        criterion = RMSELoss()
        return criterion


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
