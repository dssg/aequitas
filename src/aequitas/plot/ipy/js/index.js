import React from "react";
import ReactDOM from "react-dom";
import { select } from "d3-selection";

import "./index.scss";

import Disparity from "~/charts/disparity";

export function plotDisparityBubbleChart(divId, payload) {
  ReactDOM.render(
    <Disparity
      metrics={payload.metrics}
      attribute={payload.attribute}
      data={payload.data}
      accessibilityMode={payload["accessibility_mode"]}
      fairnessThreshold={payload["fairness_threshold"]}
    />,
    select(divId).node()
  );
}
