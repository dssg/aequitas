import React, { useState } from "react";

import sizes from "~/constants/sizes";

const INPUT_IDS = {
  LOWER: "lowerBoundInput",
  UPPER: "upperBoundInput"
};

const validateInputWithRegex = (value) => {
  // check if target value passes regex (float with .01 precision) & is within appropriate bounds
  const re = /[+]?([0-1]*[.])?[0-9]+/;
  return re.test(value) && value >= 0 && value <= 1;
};

export default function AxisInputs(props) {
  // These are in the JS so that we can use them for placing the foreignObjects
  const inputStyles = {
    width: sizes.BOUNDS_INPUT.width,
    height: sizes.BOUNDS_INPUT.height
  };
  // To center the inputs in the axis vertical space, so they overlap and align with the axis labels
  const inputYPosition =
    sizes.AXIS.TOP.height / 2 - sizes.BOUNDS_INPUT.height / 2;
  // To remove the autocomplete of the inputs
  const onFocus = (event) => {
    event.target.setAttribute("autocomplete", "off");
  };
  const [invalidLowerBound, setInvalidLowerBound] = useState(false);
  const [invalidUpperBound, setInvalidUpperBound] = useState(false);
  const setInvalidBound = {};
  setInvalidBound[INPUT_IDS.LOWER] = setInvalidLowerBound;
  setInvalidBound[INPUT_IDS.UPPER] = setInvalidUpperBound;

  const handleKeyDown = (event) => {
    if (event.key !== "Enter") {
      return;
    }

    const inputId = event.target.id;
    const inputValue = event.target.value;

    if (!validateInputWithRegex(inputValue)) {
      setInvalidBound[inputId](true);
      return;
    }

    const newValue =
      Math.round((Number(inputValue) + Number.EPSILON) * 100) / 100;

    if (inputId === INPUT_IDS.LOWER && newValue < props.axisBounds[1]) {
      props.setAxisBounds([newValue, props.axisBounds[1]]);
      setInvalidUpperBound(false);
      return;
    }

    if (inputId === INPUT_IDS.UPPER && newValue > props.axisBounds[0]) {
      props.setAxisBounds([props.axisBounds[0], newValue]);
      setInvalidUpperBound(false);
      return;
    }

    setInvalidBound[inputId](true);
  };

  return (
    <>
      <foreignObject
        key={`startInput-${props.axisBounds[0]}`}
        x={sizes.AXIS.LEFT.width - sizes.BOUNDS_INPUT.width / 2}
        y={inputYPosition}
        /* Below, the * 1.2 is required so the inputs don't have their bounds cut */
        width={sizes.BOUNDS_INPUT.width * 1.2}
        height={sizes.BOUNDS_INPUT.height * 1.2}
      >
        <input
          id={INPUT_IDS.LOWER}
          className="aequitas-axis-input"
          type="text"
          defaultValue={props.axisBounds[0]}
          onKeyDown={handleKeyDown}
          onClick={(event) => event.target.select()}
          onFocus={onFocus}
          style={{
            ...inputStyles,
            border: invalidLowerBound ? "2px solid red" : "1px solid black"
          }}
        />
      </foreignObject>
      <foreignObject
        x={
          sizes.CHART_WIDTH -
          sizes.CHART_PADDING.right -
          sizes.BOUNDS_INPUT.width / 2
        }
        y={inputYPosition}
        width={sizes.BOUNDS_INPUT.width * 1.2}
        height={sizes.BOUNDS_INPUT.height * 1.2}
      >
        <input
          id={INPUT_IDS.UPPER}
          className="aequitas-axis-input"
          type="text"
          defaultValue={props.axisBounds[1]}
          onKeyDown={handleKeyDown}
          onClick={(event) => event.target.select()}
          onFocus={onFocus}
          style={{
            ...inputStyles,
            border: invalidUpperBound ? "2px solid red" : "1px solid black"
          }}
        />
      </foreignObject>
    </>
  );
}
