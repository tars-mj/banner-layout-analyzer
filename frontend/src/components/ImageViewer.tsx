import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Image, Line, Rect, Text } from 'react-konva';
import { Box, Typography, Button } from '@mui/material';
import { DetectionResult, BannerForm } from '../types/detection';
import { KonvaEventObject } from 'konva/lib/Node';
import { Stage as StageType } from 'konva/lib/Stage';
import { Text as TextType } from 'konva/lib/shapes/Text';
import { Rect as RectType } from 'konva/lib/shapes/Rect';
import { Node, NodeConfig } from 'konva/lib/Node';

interface ImageViewerProps {
    imageUrl: string;
    detections: DetectionResult;
    formData: BannerForm;
    onSectionPositionsChange: (positions: number[]) => void;
    showBoundingBoxes: boolean;
}

interface ContextMenuType {
    mouseX: number;
    mouseY: number;
    isLine: boolean;
    lineIndex?: number;
}

interface KonvaNodeWithAttrs extends Node {
    attrs: NodeConfig & {
        name?: string;
        id?: string;
    };
}

const MAX_WIDTH = 1200;
const MAX_HEIGHT = 600;

// Helper function to check if line intersects with a bounding box
const checkLineBoxCollision = (
    lineX: number,
    boxX1: number,
    boxX2: number,
    margin: number = 5
): boolean => {
    return lineX >= boxX1 - margin && lineX <= boxX2 + margin;
};

// Helper function to calculate section positions
const calculateSectionPositions = (totalWidth: number, maxSectionWidth: number, pixelsPerCm: number): number[] => {
    const positions: number[] = [];
    const maxSectionWidthPx = maxSectionWidth * pixelsPerCm;
    let remainingWidth = totalWidth;
    let currentPosition = 0;

    while (remainingWidth > maxSectionWidthPx) {
        currentPosition += maxSectionWidthPx;
        positions.push(currentPosition);
        remainingWidth -= maxSectionWidthPx;
    }

    return positions;
};

// Helper function to check if margin area collides with any bounding box
const checkMarginCollision = (
    marginRect: { x: number, y: number, width: number, height: number },
    boxes: Array<{ bbox: [number, number, number, number] }>,
    scale: number
): boolean => {
    return boxes.some(box => {
        const boxX = box.bbox[0] * scale;
        const boxY = box.bbox[1] * scale;
        const boxWidth = (box.bbox[2] - box.bbox[0]) * scale;
        const boxHeight = (box.bbox[3] - box.bbox[1]) * scale;

        return !(
            marginRect.x + marginRect.width < boxX ||
            marginRect.x > boxX + boxWidth ||
            marginRect.y + marginRect.height < boxY ||
            marginRect.y > boxY + boxHeight
        );
    });
};

export const ImageViewer = ({ 
    imageUrl, 
    detections, 
    formData,
    onSectionPositionsChange,
    showBoundingBoxes
}: ImageViewerProps) => {
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [image, setImage] = useState<HTMLImageElement | null>(null);
    const [scale, setScale] = useState(1);
    const [sectionCollisions, setSectionCollisions] = useState<boolean[]>([]);
    const [contextMenu, setContextMenu] = useState<ContextMenuType | null>(null);
    const stageRef = useRef<StageType | null>(null);
    const [debug, setDebug] = useState<string>('');
    const [isDragging] = useState(false);

    useEffect(() => {
        console.log('ImageViewer mounted');
        console.log('Initial imageUrl:', imageUrl);
        console.log('Initial detections:', detections);
        console.log('Initial dimensions:', dimensions);
        console.log('Initial scale:', scale);

        if (!imageUrl) {
            console.error('No image URL provided');
            setDebug('Error: No image URL provided');
            return;
        }

        const img = new window.Image();
        img.crossOrigin = 'anonymous';
        
        img.onload = () => {
            console.log('Image loaded successfully');
            console.log('Original dimensions:', img.width, 'x', img.height);
            console.log('Image object:', img);
            
            // Calculate scale to fit within MAX_WIDTH x MAX_HEIGHT while maintaining aspect ratio
            const scaleX = MAX_WIDTH / img.width;
            const scaleY = MAX_HEIGHT / img.height;
            const newScale = Math.min(scaleX, scaleY, 1); // Don't scale up images

            // Calculate new dimensions
            const newWidth = Math.round(img.width * newScale);
            const newHeight = Math.round(img.height * newScale);

            console.log('New dimensions:', newWidth, 'x', newHeight);
            console.log('Scale:', newScale);

            // Set image first
            setImage(img);
            console.log('Image state set');
            
            // Then set dimensions
            setDimensions({
                width: newWidth,
                height: newHeight
            });
            console.log('Dimensions state set');
            
            // Finally set scale
            setScale(newScale);
            console.log('Scale state set');
            
            setDebug(`Image loaded: ${newWidth}x${newHeight} (scale: ${newScale})`);

            // Calculate initial section positions
            if (formData.width && formData.maxSectionWidth) {
                const pixelsPerCm = newWidth / formData.width;
                const positions = calculateSectionPositions(newWidth, formData.maxSectionWidth, pixelsPerCm);
                onSectionPositionsChange(positions);
            }
        };

        img.onerror = (error) => {
            console.error('Error loading image:', error);
            console.error('Image URL that failed:', imageUrl);
            setDebug(`Error loading image: ${error}`);
        };

        console.log('Setting image source to:', imageUrl);
        img.src = imageUrl;

        // Cleanup function
        return () => {
            console.log('ImageViewer unmounting');
            if (img) {
                img.onload = null;
                img.onerror = null;
            }
        };
    }, [imageUrl, formData.width, formData.maxSectionWidth]);

    // Check collisions for all sections
    useEffect(() => {
        if (!formData.sectionPositions) return;

        const collisions = formData.sectionPositions.map(x => {
            // Check collisions with faces
            const faceCollision = detections.faces.some(box => 
                checkLineBoxCollision(x, box.bbox[0] * scale, box.bbox[2] * scale)
            );

            // Check collisions with logos
            const logoCollision = detections.logos.some(box => 
                checkLineBoxCollision(x, box.bbox[0] * scale, box.bbox[2] * scale)
            );

            // Check collisions with QR codes
            const qrCollision = detections.qrcodes.some(box => 
                checkLineBoxCollision(x, box.bbox[0] * scale, box.bbox[2] * scale)
            );

            return faceCollision || logoCollision || qrCollision;
        });

        setSectionCollisions(collisions);
    }, [formData.sectionPositions, detections, scale]);

    const handleDragMove = (e: KonvaEventObject<DragEvent>, index: number) => {
        const newX = e.target.x();
        const newPositions = [...formData.sectionPositions];
        newPositions[index] = newX;
        onSectionPositionsChange(newPositions);
    };

    const handleContextMenu = (e: KonvaEventObject<MouseEvent>) => {
        e.evt.preventDefault();
        const stage = stageRef.current;
        if (!stage) return;

        const pos = stage.getPointerPosition();
        if (!pos) return;

        // Check if click was on a line
        const clickedLine = stage.getIntersection(pos) as KonvaNodeWithAttrs;
        const isLine = clickedLine?.attrs?.name === 'section-line';
        const lineIndex = isLine ? parseInt(clickedLine?.attrs?.id || '-1') : undefined;

        setContextMenu({
            mouseX: e.evt.clientX,
            mouseY: e.evt.clientY,
            isLine: isLine,
            lineIndex: lineIndex
        });
    };

    const handleCloseContextMenu = () => {
        setContextMenu(null);
    };

    const handleAddSection = () => {
        if (!contextMenu) return;

        // Get stage element using ref
        const stage = stageRef.current;
        if (!stage) return;

        const pointerPosition = stage.getPointerPosition();
        if (!pointerPosition) return;

        const newX = pointerPosition.x;
        const newPositions = [...formData.sectionPositions];
        newPositions.push(newX);
        newPositions.sort((a, b) => a - b); // Sort positions to keep lines in correct order
        onSectionPositionsChange(newPositions);
        handleCloseContextMenu();
    };

    const handleLineContextMenu = (e: KonvaEventObject<MouseEvent>, index: number) => {
        e.evt.preventDefault();
        e.evt.stopPropagation();
        
        setContextMenu({
            mouseX: e.evt.clientX,
            mouseY: e.evt.clientY,
            isLine: true,
            lineIndex: index
        });
    };

    const handleDeleteSection = () => {
        if (!contextMenu?.isLine || contextMenu.lineIndex === undefined) return;

        const newPositions = formData.sectionPositions.filter((_, i) => i !== contextMenu.lineIndex);
        onSectionPositionsChange(newPositions);
        handleCloseContextMenu();
    };

    if (!image) {
        console.log('No image object available');
        return (
            <div style={{ 
                padding: 20,
                textAlign: 'center',
                backgroundColor: '#fff',
                border: '1px dashed #ccc',
                borderRadius: '4px',
                margin: '10px'
            }}>
                <p>Loading image... {debug}</p>
                <p>Image URL: {imageUrl}</p>
                <p>Current dimensions: {dimensions.width} x {dimensions.height}</p>
                <p>Current scale: {scale}</p>
            </div>
        );
    }

    const renderSections = () => {
        if (!formData.showSections || !formData.sectionPositions) return null;

        return (
            <>
                {formData.sectionPositions.map((x, i) => {
                    // Calculate absolute position in cm
                    const positionInCm = Math.round((x / dimensions.width) * formData.width);

                    return (
                        <React.Fragment key={`section-${i}`}>
                            <Line
                                x={x}
                                y={0}
                                points={[0, 0, 0, dimensions.height]}
                                stroke={sectionCollisions[i] ? "#FF0000" : "#00FF00"}
                                strokeWidth={4}
                                dash={[10, 10]}
                                draggable
                                name="section-line"
                                id={`${i}`}
                                onDragMove={(e) => handleDragMove(e, i)}
                                dragBoundFunc={(pos) => ({
                                    x: Math.max(
                                        i === 0 ? 0 : formData.sectionPositions[i - 1] + 10,
                                        Math.min(
                                            pos.x,
                                            i === formData.sectionPositions.length - 1 
                                                ? dimensions.width 
                                                : formData.sectionPositions[i + 1] - 10
                                        )
                                    ),
                                    y: 0
                                })}
                                onMouseEnter={(e) => {
                                    const container = e.target.getStage()?.container();
                                    if (container) {
                                        container.style.cursor = 'pointer';
                                    }
                                }}
                                onMouseLeave={(e) => {
                                    const container = e.target.getStage()?.container();
                                    if (container) {
                                        container.style.cursor = 'default';
                                    }
                                }}
                                onDragStart={(e) => {
                                    const container = e.target.getStage()?.container();
                                    if (container) {
                                        container.style.cursor = 'grabbing';
                                    }
                                }}
                                onDragEnd={(e) => {
                                    const container = e.target.getStage()?.container();
                                    if (container) {
                                        container.style.cursor = 'pointer';
                                    }
                                }}
                                onContextMenu={(e) => handleLineContextMenu(e, i)}
                            />
                            <Text
                                x={x - 20}
                                y={10}
                                text={`${positionInCm}cm`}
                                fill={sectionCollisions[i] ? "#FF0000" : "#00FF00"}
                                fontSize={12}
                            />
                        </React.Fragment>
                    );
                })}
            </>
        );
    };

    const renderMargins = () => {
        if (!formData.showMargins || formData.margin <= 0) return null;

        const marginPx = (formData.margin / formData.width) * dimensions.width;
        const allBoxes = [...detections.faces, ...detections.logos, ...detections.qrcodes];

        // Top margin
        const topMargin = {
            x: 0,
            y: 0,
            width: dimensions.width,
            height: marginPx
        };
        const topCollision = checkMarginCollision(topMargin, allBoxes, scale);

        // Bottom margin
        const bottomMargin = {
            x: 0,
            y: dimensions.height - marginPx,
            width: dimensions.width,
            height: marginPx
        };
        const bottomCollision = checkMarginCollision(bottomMargin, allBoxes, scale);

        // Left margin
        const leftMargin = {
            x: 0,
            y: 0,
            width: marginPx,
            height: dimensions.height
        };
        const leftCollision = checkMarginCollision(leftMargin, allBoxes, scale);

        // Right margin
        const rightMargin = {
            x: dimensions.width - marginPx,
            y: 0,
            width: marginPx,
            height: dimensions.height
        };
        const rightCollision = checkMarginCollision(rightMargin, allBoxes, scale);

        return (
            <>
                {/* Top margin */}
                <Rect
                    {...topMargin}
                    fill={topCollision ? '#FF000066' : '#00FF0033'}
                />
                {/* Bottom margin */}
                <Rect
                    {...bottomMargin}
                    fill={bottomCollision ? '#FF000066' : '#00FF0033'}
                />
                {/* Left margin */}
                <Rect
                    {...leftMargin}
                    fill={leftCollision ? '#FF000066' : '#00FF0033'}
                />
                {/* Right margin */}
                <Rect
                    {...rightMargin}
                    fill={rightCollision ? '#FF000066' : '#00FF0033'}
                />
            </>
        );
    };

    const renderDetections = () => {
        if (!showBoundingBoxes) return null;

        return (
            <>
                {detections.faces.map((box, i) => (
                    <React.Fragment key={`face-${i}`}>
                        <Rect
                            x={box.bbox[0] * scale}
                            y={box.bbox[1] * scale}
                            width={(box.bbox[2] - box.bbox[0]) * scale}
                            height={(box.bbox[3] - box.bbox[1]) * scale}
                            stroke="#0000FF"
                            strokeWidth={2}
                            dash={[2, 2]}
                            fill="transparent"
                            onMouseEnter={(e: KonvaEventObject<MouseEvent>) => {
                                const rect = e.target as RectType;
                                rect.fill('#0000FF66');
                                const text = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `text-face-${i}`
                                ) as TextType;
                                if (text) {
                                    text.visible(true);
                                    rect.getLayer()?.batchDraw();
                                }
                            }}
                            onMouseLeave={(e: KonvaEventObject<MouseEvent>) => {
                                const rect = e.target as RectType;
                                rect.fill('transparent');
                                const text = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `text-face-${i}`
                                ) as TextType;
                                if (text) {
                                    text.visible(false);
                                    rect.getLayer()?.batchDraw();
                                }
                            }}
                        />
                        <Text
                            id={`text-face-${i}`}
                            x={box.bbox[0] * scale + ((box.bbox[2] - box.bbox[0]) * scale / 2)}
                            y={box.bbox[1] * scale + ((box.bbox[3] - box.bbox[1]) * scale / 2)}
                            text={`face (${Math.round(box.confidence * 100)}%)`}
                            fill="white"
                            fontSize={16}
                            visible={false}
                            offsetX={50}
                            offsetY={10}
                            listening={false}
                            padding={5}
                            background="#000000CC"
                        />
                    </React.Fragment>
                ))}
                {detections.logos.map((box, i) => (
                    <React.Fragment key={`logo-${i}`}>
                        <Rect
                            x={box.bbox[0] * scale}
                            y={box.bbox[1] * scale}
                            width={(box.bbox[2] - box.bbox[0]) * scale}
                            height={(box.bbox[3] - box.bbox[1]) * scale}
                            stroke="#0000FF"
                            strokeWidth={2}
                            dash={[2, 2]}
                            fill="transparent"
                            onMouseEnter={(e: KonvaEventObject<MouseEvent>) => {
                                const rect = e.target as RectType;
                                rect.fill('#0000FF66');
                                const text = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `text-logo-${i}`
                                ) as TextType;
                                if (text) {
                                    text.visible(true);
                                    rect.getLayer()?.batchDraw();
                                }
                            }}
                            onMouseLeave={(e: KonvaEventObject<MouseEvent>) => {
                                const rect = e.target as RectType;
                                rect.fill('transparent');
                                const text = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `text-logo-${i}`
                                ) as TextType;
                                if (text) {
                                    text.visible(false);
                                    rect.getLayer()?.batchDraw();
                                }
                            }}
                        />
                        <Text
                            id={`text-logo-${i}`}
                            x={box.bbox[0] * scale + ((box.bbox[2] - box.bbox[0]) * scale / 2)}
                            y={box.bbox[1] * scale + ((box.bbox[3] - box.bbox[1]) * scale / 2)}
                            text={`logo (${Math.round(box.confidence * 100)}%)`}
                            fill="white"
                            fontSize={16}
                            visible={false}
                            offsetX={50}
                            offsetY={10}
                            listening={false}
                            padding={5}
                            background="#000000CC"
                        />
                    </React.Fragment>
                ))}
                {detections.qrcodes.map((box, i) => (
                    <React.Fragment key={`qr-${i}`}>
                        <Rect
                            x={box.bbox[0] * scale}
                            y={box.bbox[1] * scale}
                            width={(box.bbox[2] - box.bbox[0]) * scale}
                            height={(box.bbox[3] - box.bbox[1]) * scale}
                            stroke="#0000FF"
                            strokeWidth={2}
                            dash={[2, 2]}
                            fill="transparent"
                            onMouseEnter={(e: KonvaEventObject<MouseEvent>) => {
                                const rect = e.target as RectType;
                                rect.fill('#0000FF66');
                                const text = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `text-qr-${i}`
                                ) as TextType;
                                if (text) {
                                    text.visible(true);
                                }
                                const status = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `status-qr-${i}`
                                ) as TextType;
                                if (status) {
                                    status.visible(true);
                                }
                                rect.getLayer()?.batchDraw();
                            }}
                            onMouseLeave={(e: KonvaEventObject<MouseEvent>) => {
                                const rect = e.target as RectType;
                                rect.fill('transparent');
                                const text = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `text-qr-${i}`
                                ) as TextType;
                                if (text) {
                                    text.visible(false);
                                }
                                const status = rect.getLayer()?.findOne((node: Node) => 
                                    node.attrs.id === `status-qr-${i}`
                                ) as TextType;
                                if (status) {
                                    status.visible(false);
                                }
                                rect.getLayer()?.batchDraw();
                            }}
                        />
                        <Text
                            id={`text-qr-${i}`}
                            x={box.bbox[0] * scale + ((box.bbox[2] - box.bbox[0]) * scale / 2)}
                            y={box.bbox[1] * scale + ((box.bbox[3] - box.bbox[1]) * scale / 2)}
                            text="QR Code"
                            fill="white"
                            fontSize={16}
                            visible={false}
                            offsetX={35}
                            offsetY={-25}
                            listening={false}
                            padding={5}
                            background="#000000CC"
                            align="center"
                        />
                        <Text
                            id={`status-qr-${i}`}
                            x={box.bbox[0] * scale + ((box.bbox[2] - box.bbox[0]) * scale / 2)}
                            y={box.bbox[1] * scale + ((box.bbox[3] - box.bbox[1]) * scale / 2)}
                            text={box.isValidUrl ? "✓" : "✗"}
                            fill={box.isValidUrl ? "#00FF00" : "#FF0000"}
                            fontSize={48}
                            visible={false}
                            offsetX={12}
                            offsetY={25}
                            fontStyle="bold"
                            listening={false}
                            align="center"
                        />
                    </React.Fragment>
                ))}
            </>
        );
    };

    // Add debug logging for main render
    console.log('Rendering ImageViewer with:', {
        image: !!image,
        dimensions,
        scale,
        showBoundingBoxes,
        'detections.qrcodes.length': detections.qrcodes.length
    });

    const stageWidth = dimensions.width;
    const stageHeight = dimensions.height;

    return (
        <Box sx={{ position: 'relative', width: 'fit-content' }}>
            <Stage
                width={stageWidth}
                height={stageHeight}
                onContextMenu={handleContextMenu}
                onClick={handleCloseContextMenu}
                style={{ cursor: isDragging ? 'grabbing' : 'default' }}
                ref={stageRef}
            >
                <Layer>
                    {/* Background image */}
                    <Image
                        image={image}
                        width={stageWidth}
                        height={stageHeight}
                    />
                    
                    {/* Render detections first, then margins and sections */}
                    {showBoundingBoxes && renderDetections()}
                    {formData.showMargins && renderMargins()}
                    {formData.showSections && renderSections()}
                </Layer>
            </Stage>

            {/* Section width measurements */}
            {formData.showSections && formData.sectionPositions.length > 0 && (
                <Box sx={{ 
                    width: `${stageWidth}px`, 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    mt: 1,
                    position: 'relative',
                    height: '40px',  // Height for the lines
                    boxSizing: 'border-box'
                }}>
                    {/* Horizontal connecting line */}
                    <div style={{
                        position: 'absolute',
                        top: '20px',
                        left: '0',
                        right: '0',
                        height: '1px',
                        backgroundColor: '#666'
                    }} />
                    
                    {/* Vertical lines at the ends and at the divisions */}
                    <div style={{
                        position: 'absolute',
                        top: '15px',
                        left: '0',
                        width: '1px',
                        height: '10px',
                        backgroundColor: '#666'
                    }} />
                    
                    {formData.sectionPositions.map((pos, index) => (
                        <div
                            key={`line-${index}`}
                            style={{
                                position: 'absolute',
                                top: '15px',
                                left: `${pos}px`,
                                width: '1px',
                                height: '10px',
                                backgroundColor: '#666'
                            }}
                        />
                    ))}
                    
                    <div style={{
                        position: 'absolute',
                        top: '15px',
                        right: '0',
                        width: '1px',
                        height: '10px',
                        backgroundColor: '#666'
                    }} />

                    {formData.sectionPositions.map((pos, index) => {
                        if (index === 0) {
                            // First section - from start to first line
                            const widthInCm = Math.round((pos / stageWidth) * formData.width);
                            return (
                                <Typography 
                                    key={`width-${index}`}
                                    variant="body2"
                                    sx={{ 
                                        position: 'absolute',
                                        left: pos / 2,
                                        transform: 'translateX(-50%)',
                                        top: 0
                                    }}
                                >
                                    {widthInCm} cm
                                </Typography>
                            );
                        } else {
                            // Calculate width between current and previous line
                            const prevPos = formData.sectionPositions[index - 1];
                            const widthInCm = Math.round(((pos - prevPos) / stageWidth) * formData.width);
                            return (
                                <Typography 
                                    key={`width-${index}`}
                                    variant="body2"
                                    sx={{ 
                                        position: 'absolute',
                                        left: prevPos + (pos - prevPos) / 2,
                                        transform: 'translateX(-50%)',
                                        top: 0
                                    }}
                                >
                                    {widthInCm} cm
                                </Typography>
                            );
                        }
                    })}
                    {/* Last section - from last line to end */}
                    {formData.sectionPositions.length > 0 && (
                        <Typography 
                            variant="body2"
                            sx={{ 
                                position: 'absolute',
                                left: (stageWidth + formData.sectionPositions[formData.sectionPositions.length - 1]) / 2,
                                transform: 'translateX(-50%)',
                                top: 0
                            }}
                        >
                            {Math.round(((stageWidth - formData.sectionPositions[formData.sectionPositions.length - 1]) / stageWidth) * formData.width)} cm
                        </Typography>
                    )}
                </Box>
            )}

            {/* Context menu */}
            {contextMenu !== null && (
                <div
                    style={{
                        position: 'fixed',
                        top: contextMenu.mouseY - 5,  // Move the menu up
                        left: contextMenu.mouseX - 5,  // Move the menu left
                        zIndex: 1000,
                        backgroundColor: 'white',
                        boxShadow: '0px 0px 10px rgba(0,0,0,0.1)',
                        borderRadius: '4px',
                        padding: '8px'
                    }}
                    onClick={(e) => {
                        e.stopPropagation();
                    }}
                >
                    {(!contextMenu.isLine || contextMenu.lineIndex === undefined) && (
                        <Button
                            size="small"
                            onClick={(e) => {
                                e.stopPropagation();
                                handleAddSection();
                            }}
                            // sx={{ display: 'block', mb: 1 }}
                        >
                            Add division line
                        </Button>
                    )}
                    {contextMenu.isLine && contextMenu.lineIndex !== undefined && (
                        <Button
                            size="small"
                            onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteSection();
                            }}
                            color="error"
                        >
                            Delete line
                        </Button>
                    )}
                </div>
            )}
        </Box>
    );
};