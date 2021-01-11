import React from "react";
import ReactDOM from "react-dom";
import { select } from "d3-selection";

import "./index.scss";

import DisparityLayout from "./containers/DisparityLayout";

export function plotDisparityBubbleChart(divId, payload) {
  ReactDOM.render(
    <DisparityLayout
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
