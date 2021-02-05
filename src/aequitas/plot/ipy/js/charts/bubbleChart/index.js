import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import Select from "react-select";
import { isNull } from "lodash";

import Chart from "./chart";
import AxisDisparity from "./chart/AxisDisparity";
import AxisAbsolute from "./chart/AxisAbsolute";
import ThresholdsAbsolute from "./chart/ThresholdsAbsolute";
import ThresholdsDisparity from "./chart/ThresholdsDisparity";
import Legend from "~/components/Legend";
import Footnote from "~/components/Footnote";

import {
  getScaleColor,
  getScaleShape,
  getScaleSizeBubble,
  getScalePositionDisparity,
  getScalePositionAbsolute
} from "~/utils/scales";
import { toTitleCase } from "~/utils/helpers";
import sizes from "~/constants/sizes";
import colors from "~/constants/colors.scss";

import "./style.scss";

const propTypes = {
  isDisparityChart: PropTypes.bool.isRequired,
  metrics: PropTypes.array.isRequired,
  attribute: PropTypes.string.isRequired,
  data: PropTypes.array.isRequired,
  accessibilityMode: PropTypes.bool.isRequired,
  fairnessThreshold: PropTypes.number
};

function BubbleChart(props) {
  //
  // ATTRIBUTES
  //
  const attributes = [
    ...new Set(props.data.map((row) => row["attribute_name"]))
  ];

  //
  // HOOKS: STATE & EFFECT
  //
  const [activeGroup, setActiveGroup] = useState(null);
  const [selectedAttribute, setSelectedAttribute] = useState(
    props.attribute || attributes[0]
  );
  const [dataToPlot, setDataToPlot] = useState(
    props.data.filter((row) => row["attribute_name"] === selectedAttribute)
  );
  const [referenceGroup, setReferenceGroup] = useState(
    dataToPlot[0][`${props.metrics[0]}_ref_group_value`]
  );

  useEffect(() => {
    const newData = props.data.filter(
      (row) => row["attribute_name"] === selectedAttribute
    );
    setDataToPlot(newData);
    setReferenceGroup(newData[0][`${props.metrics[0]}_ref_group_value`]);
  }, [props.data, props.metrics, selectedAttribute]);

  //
  //   VARIABLES
  //
  const selectOptions = attributes.map((attribute) => {
    return {
      value: attribute,
      label: toTitleCase(attribute)
    };
  });
  const groups = [
    referenceGroup,
    ...dataToPlot
      .map((row) => row["attribute_value"])
      .filter((item) => item !== referenceGroup)
  ];
  const chartAreaHeight =
    sizes.ROW_HEIGHT * props.metrics.length + sizes.MARGIN.top;
  const dataColumnSuffix = props.isDisparityChart ? "_disparity_scaled" : "";
  let dataColumnNames = {};
  props.metrics.map(
    (metric) => (dataColumnNames[metric] = `${metric}${dataColumnSuffix}`)
  );

  //
  // SCALES
  //
  const scaleColor = getScaleColor(groups);
  const scaleShape = getScaleShape(groups, props.accessibilityMode);
  const scaleBubbleSize = getScaleSizeBubble(props.data);
  const scalePosition = props.isDisparityChart
    ? getScalePositionDisparity(
        props.data,
        props.metrics,
        props.fairnessThreshold
      )
    : getScalePositionAbsolute();

  //
  // DISPARITY/ABSOLUTE COMPONENTS
  //
  const Axis = props.isDisparityChart ? AxisDisparity : AxisAbsolute;
  const AxisComponent = (
    <Axis scale={scalePosition} chartAreaHeight={chartAreaHeight} />
  );

  let ThresholdsComponent, Thresholds;
  if (!isNull(props.fairnessThreshold)) {
    Thresholds = props.isDisparityChart
      ? ThresholdsDisparity
      : ThresholdsAbsolute;
    ThresholdsComponent = (
      <Thresholds
        fairnessThreshold={props.fairnessThreshold}
        scalePosition={scalePosition}
        thresholdColor={
          props.accessibilityMode ? colors.referenceGrey : colors.thresholdRed
        }
        chartAreaHeight={chartAreaHeight}
        metrics={props.metrics}
        referenceGroup={referenceGroup}
        data={dataToPlot}
      />
    );
  } else {
    ThresholdsComponent = null;
  }

  //
  // RETURN
  //
  return (
    <div className="aequitas">
      <div className="aequitas-title">
        <h1>
          {props.isDisparityChart ? "Disparities" : "Absolute Values"} on{" "}
        </h1>
        <Select
          className="aequitas-select"
          classNamePrefix="aequitas-select"
          options={selectOptions}
          defaultValue={selectOptions[0]}
          onChange={(option) => setSelectedAttribute(option.value)}
        />
      </div>
      <div className="aequitas-chart-area">
        <Chart
          data={dataToPlot}
          handleActiveGroup={setActiveGroup}
          metrics={props.metrics}
          referenceGroup={referenceGroup}
          scaleColor={scaleColor}
          scaleShape={scaleShape}
          scaleBubbleSize={scaleBubbleSize}
          scalePosition={scalePosition}
          activeGroup={activeGroup}
          AxisComponent={AxisComponent}
          ThresholdsComponent={ThresholdsComponent}
          chartAreaHeight={chartAreaHeight}
          dataColumnNames={dataColumnNames}
        />
        <Legend
          groups={groups}
          activeGroup={activeGroup}
          handleSelect={setActiveGroup}
          scaleColor={scaleColor}
          scaleShape={scaleShape}
        />
      </div>
      <Footnote
        fairnessThreshold={props.fairnessThreshold}
        referenceGroup={referenceGroup}
        accessibilityMode={props.accessibilityMode}
      />
    </div>
  );
}

BubbleChart.propTypes = propTypes;

export default BubbleChart;
