import React from "react";
import ReactDOM from "react-dom";
import { select } from "d3-selection";

import "./index.scss";

import DisparityChart from "./containers/DisparityChart";
import DisparityChartLayout from "./containers/DisparityChartLayout";

export function plotDisparityBubbleChart(divId, payload) {
  ReactDOM.render(
    <DisparityChartLayout
      metrics={payload.metrics}
      attribute={payload.attribute}
      referenceGroup={payload["ref_group"]}
      data={payload.data}
      accessibilityMode={payload["accessibility_mode"]}
      fairnessThreshold={payload["fairness_threshold"]}
    />,
    select(divId).node()
  );
}
