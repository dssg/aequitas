import React from "react";
import ReactDOM from "react-dom";
import { select } from "d3-selection";

import "./index.scss";

import DisparityChart from "./containers/DisparityChart";

export function plotDisparityBubbleChart(divId, payload) {
  ReactDOM.render(
    <DisparityChart
      metrics={payload.metrics}
      attribute={payload.attribute}
      data={payload.data}
    />,
    select(divId).node()
  );
}
