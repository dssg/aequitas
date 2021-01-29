import React from "react";
import ReactDOM from "react-dom";
import { select } from "d3-selection";

import "./index.scss";

import BubbleChart from "~/charts/bubbleChart";

export function plotDisparityBubbleChart(divId, payload) {
  ReactDOM.render(
    <BubbleChart
      isDisparityChart={true}
      metrics={payload.metrics}
      attribute={payload.attribute}
      data={payload.data}
      accessibilityMode={payload["accessibility_mode"]}
      fairnessThreshold={payload["fairness_threshold"]}
    />,
    select(divId).node()
  );
}

export function plotMetricBubbleChart(divId, payload) {
  ReactDOM.render(
    <BubbleChart
      isDisparityChart={false}
      metrics={payload.metrics}
      attribute={payload.attribute}
      data={payload.data}
      accessibilityMode={payload["accessibility_mode"]}
      fairnessThreshold={payload["fairness_threshold"]}
    />,
    select(divId).node()
  );
}
