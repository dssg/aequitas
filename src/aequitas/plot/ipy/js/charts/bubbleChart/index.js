import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import Select from "react-select";

import Chart from "./chart";
import Legend from "~/components/Legend";
import Footnote from "~/components/Footnote";

import { getScaleColor, getScaleShape } from "~/utils/scales";
import { toTitleCase } from "~/utils/helpers";
import "./style.scss";

const propTypes = {
  isDisparityChart: PropTypes.bool.isRequired,
  metrics: PropTypes.array.isRequired,
  attribute: PropTypes.string.isRequired,
  data: PropTypes.array.isRequired,
  accessibilityMode: PropTypes.bool.isRequired,
  fairnessThreshold: PropTypes.number.isRequired
};

function BubbleChart(props) {
  const attributes = [
    ...new Set(props.data.map((row) => row["attribute_name"]))
  ];

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

  const scaleColor = getScaleColor(groups);
  const scaleShape = getScaleShape(groups, props.accessibilityMode);

  useEffect(() => {
    setDataToPlot(
      props.data.filter((row) => row["attribute_name"] === selectedAttribute)
    );
  }, [props.data, selectedAttribute]);

  useEffect(() => {
    setReferenceGroup(dataToPlot[0][`${props.metrics[0]}_ref_group_value`]);
  }, [props.metrics, dataToPlot]);

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
          accessibilityMode={props.accessibilityMode}
          isDisparityChart={props.isDisparityChart}
          data={dataToPlot}
          fairnessThreshold={props.fairnessThreshold}
          handleActiveGroup={setActiveGroup}
          metrics={props.metrics}
          referenceGroup={referenceGroup}
          scaleColor={scaleColor}
          scaleShape={scaleShape}
          activeGroup={activeGroup}
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
