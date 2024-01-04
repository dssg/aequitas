/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2022 Feedzai, Strictly Confidential
 */

 import React from "react";
 import PropTypes from "prop-types";
 import { Box, Slider, Stack } from "@mui/material";
 import { lightGrey } from "./RangeSlider.scss";
 import { formatPercentage } from "~/utils/formatters";

 
 function RangeSlider({ domainRange, handleDomainChange, isHorizontal, dimensions, otherDomain, 
    setThisAxisDomain, setOtherAxisDomain, zoom }) {

  return (
    <Stack spacing={2} direction="row" sx={{ mb: 1 }} alignItems="center">
          <Box sx={isHorizontal ? { width: dimensions } : { height: dimensions }}>
            <Slider
              getAriaLabel={() => "Minimum distance"}
              sx={isHorizontal ? 
                {color: lightGrey} 
                :
                {
                  '& input[type="range"]': {
                    WebkitAppearance: 'slider-vertical',
                  },
                  color: lightGrey
                }}
              orientation={isHorizontal ? "horizontal" : "vertical"}
              min={0}
              max={1}
              step={0.01}
              value={domainRange}
              valueLabelFormat={formatPercentage}
              size="small"
              onChange={(evt) => {handleDomainChange(evt.target.value, domainRange, otherDomain, 
                  setThisAxisDomain, setOtherAxisDomain, zoom)}}
              valueLabelDisplay="auto"
              disableSwap
            />
          </Box>
    </Stack>
  );
 }

 RangeSlider.propTypes = {
  domainRange: PropTypes.array.isRequired, 
  handleDomainChange: PropTypes.func.isRequired, 
  isHorizontal: PropTypes.bool.isRequired, 
  dimensions: PropTypes.number.isRequired, 
  otherDomain: PropTypes.array.isRequired, 
  setThisAxisDomain: PropTypes.func.isRequired, 
  setOtherAxisDomain: PropTypes.func.isRequired,
  zoom: PropTypes.object.isRequired,
};
 
 export default RangeSlider;