import React from "react";
import PropTypes from "prop-types";

import ThresholdBand from "~/components/ThresholdBand";
import sizes from "~/constants/sizes";
import "./style.scss";

const propTypes = {
  thresholdColor: PropTypes.string.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  chartAreaHeight: PropTypes.number.isRequired,
  scalePosition: PropTypes.func.isRequired
};

function ThresholdsDisparity(props) {
  function getSingleThresholdBand(side) {
    let x, width, ruleX;

    if (side === "left") {
      x = props.scalePosition.range()[0];
      ruleX = props.scalePosition(-props.fairnessThreshold + 1);
      width = ruleX - x;
    } else if (side === "right") {
      x = props.scalePosition(props.fairnessThreshold - 1);
      width = props.scalePosition.range()[1] - x;
      ruleX = x;
    } else {
      return null;
    }

    return (
      <ThresholdBand
        key={`${side}`}
        thresholdColor={props.thresholdColor}
        fairnessThreshold={props.fairnessThreshold}
        svgKeySuffix={`${side}`}
        x={x}
        ruleX={ruleX}
        y1={sizes.MARGIN.top}
        y2={sizes.MARGIN.top + props.chartAreaHeight}
        width={width}
        thresholdDisplayString={props.fairnessThreshold.toString()}
      />
    );
  }

  const sides = ["left", "right"];
  return <g>{sides.map((side) => getSingleThresholdBand(side))}</g>;
}

ThresholdsDisparity.propTypes = propTypes;
export default ThresholdsDisparity;
