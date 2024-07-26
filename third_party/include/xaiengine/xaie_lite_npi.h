/******************************************************************************
* Copyright (C) 2021 - 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_lite_npi.h
* @{
*
* This header file defines a lightweight version of AIE NPI registers offsets
* and operations.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Wendy   09/06/2021  Initial creation
* </pre>
*
******************************************************************************/
#ifndef XAIE_LITE_NPI_H
#define XAIE_LITE_NPI_H

/************************** Constant Definitions *****************************/
#define XAIE_NPI_TIMEOUT_US				40U

#define XAIE_NPI_PCSR_MASK_REG				0x00000000U
#define XAIE_NPI_PCSR_MASK_SHIM_RESET_MSK		0x08000000U
#define XAIE_NPI_PCSR_MASK_SHIM_RESET_LSB		27U

#define XAIE_NPI_PCSR_CONTROL_REG			0X00000004U
#define XAIE_NPI_PCSR_CONTROL_SHIM_RESET_MSK		0x08000000U
#define XAIE_NPI_PCSR_CONTROL_SHIM_RESET_LSB		27U

#define XAIE_NPI_PCSR_LOCK_REG				0X0000000CU
#define XAIE_NPI_PCSR_LOCK_STATE_UNLOCK_CODE		0xF9E8D7C6U

#define XAIE_NPI_PROT_REG_CNTR_REG			0x00000200U
#define XAIE_NPI_PROT_REG_CNTR_EN_MSK			0x00000001U
#define XAIE_NPI_PROT_REG_CNTR_EN_LSB			0U
#define XAIE_NPI_PROT_REG_CNTR_FIRSTCOL_MSK		0x000000FEU
#define XAIE_NPI_PROT_REG_CNTR_FIRSTCOL_LSB		1U
#define XAIE_NPI_PROT_REG_CNTR_LASTCOL_MSK		0x00007F00U
#define XAIE_NPI_PROT_REG_CNTR_LASTCOL_LSB		8U

/***************************** Include Files *********************************/
#include "xaie_lite_io.h"
#include "xaiegbl_defs.h"

/************************** Variable Definitions *****************************/
/************************** Function Prototypes  *****************************/

/*****************************************************************************/
/**
*
* This is function to write value to NPI register and will not return until it
* is written to the register.
*
* @param	RegAddr: NPI register address
* @param	Val: value to write to the register
*
* @note		This function is internal.
*******************************************************************************/
static inline void _XAie_LNpiWriteCheck32(u64 RegAddr, u32 Val)
{
	_XAie_LRawWrite32((XAIE_NPI_BASEADDR + RegAddr), Val);
	_XAie_LRawPoll32((XAIE_NPI_BASEADDR + RegAddr), 0, 0, 0);
}

/*****************************************************************************/
/**
*
* This is function to write value to NPI register
*
* @param	RegAddr: NPI register address
* @param	Val: value to write to the register
*
* @note		This function is internal.
*		This function is for the NPI registers which are write only.
*******************************************************************************/
static inline void _XAie_LNpiWrite32(u64 RegAddr, u32 Val)
{
	_XAie_LRawWrite32((XAIE_NPI_BASEADDR + RegAddr), Val);
}

/*****************************************************************************/
/**
*
* This is function to set NPI lock
*
* @param	Lock : XAIE_ENABLE to lock, XAIE_DISABLE to unlock
*
* @note		This function is internal.
*******************************************************************************/
static inline void _XAie_LNpiSetLock(u8 Lock)
{
	u32 LockVal;

	if (Lock == XAIE_DISABLE) {
		LockVal = XAIE_NPI_PCSR_LOCK_STATE_UNLOCK_CODE;
	} else {
		LockVal = 0;
	}

	_XAie_LNpiWriteCheck32(XAIE_NPI_PCSR_LOCK_REG, LockVal);
}


/*****************************************************************************/
/**
*
* This is function to mask write to PCSR register
*
* @param	RegVal : Value to write to PCSR register
* @param	Mask : Mask to write to PCSR register
*
* @note		Sequence to write PCSR control register is as follows:
*		* unlock the PCSR register
*		* enable PCSR mask from mask register
*		* set the value to PCSR control register
*		* disable PCSR mask from mask register
*		* lock the PCSR register
*		This function is internal.
*******************************************************************************/
static inline void _XAie_LNpiWritePcsr(u32 RegVal, u32 Mask)
{
	_XAie_LNpiSetLock(XAIE_DISABLE);

	_XAie_LNpiWriteCheck32(XAIE_NPI_PCSR_MASK_REG, Mask);
	_XAie_LNpiWriteCheck32(XAIE_NPI_PCSR_CONTROL_REG, (RegVal & Mask));
	_XAie_LNpiWriteCheck32(XAIE_NPI_PCSR_MASK_REG, 0);

	_XAie_LNpiSetLock(XAIE_ENABLE);
}

/*****************************************************************************/
/**
*
* This is the NPI function to set the SHIM set assert
*
* @param	DevInst : AI engine device pointer
* @param	RstEnable : XAIE_ENABLE to assert reset, and XAIE_DISABLE to
*			    deassert reset.
*
* @return	XAIE_OK for success, and error value for failure
*
* @note		This function is internal.
*
*******************************************************************************/
static inline void _XAie_LNpiSetShimReset(u8 RstEnable)
{
	u32 RegVal;

	RegVal = XAie_SetField(RstEnable, XAIE_NPI_PCSR_CONTROL_SHIM_RESET_LSB,
			XAIE_NPI_PCSR_CONTROL_SHIM_RESET_MSK);

	_XAie_LNpiWritePcsr(RegVal, XAIE_NPI_PCSR_CONTROL_SHIM_RESET_MSK);
}

#endif		/* end of protection macro */
/** @} */
