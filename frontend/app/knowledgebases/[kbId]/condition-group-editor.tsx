'use client';
import React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Trash2Icon, PlusIcon } from 'lucide-react';
import { DatetimeInput } from '../datetime';
import { MetadataConfig } from '../kbconfig';

export interface MetadataCondition {
  name: string;
  comparison_operator: string;
  value: string | number;
}

export interface ConditionGroup {
  logical_operator: string;
  conditions: MetadataCondition[];
  condition_groups: ConditionGroup[];
}

export function createEmptyConditionGroup(): ConditionGroup {
  return {
    logical_operator: 'and',
    conditions: [],
    condition_groups: [],
  };
}

const MAX_DEPTH = 4;

const DEFAULT_COMPARATORS = [
  'contains', 'not contains', 'start with', 'end with',
  'is', 'is not', 'empty', 'not empty',
  '=', '≠', '>', '<', '≥', '≤',
  'in', 'not in',
  'before', 'after',
];

interface ConditionGroupEditorProps {
  group: ConditionGroup;
  onChange: (group: ConditionGroup) => void;
  onDelete?: () => void;
  metadataConfigs: MetadataConfig[];
  metadataValueTypes: { [k: string]: string };
  depth?: number;
  t: (key: string) => string;
}

export function ConditionGroupEditor({
  group,
  onChange,
  onDelete,
  metadataConfigs,
  metadataValueTypes,
  depth = 0,
  t,
}: ConditionGroupEditorProps) {
  const updateGroup = (partial: Partial<ConditionGroup>) => {
    onChange({ ...group, ...partial });
  };

  const addCondition = () => {
    updateGroup({
      conditions: [...group.conditions, { name: '', comparison_operator: '', value: '' }],
    });
  };

  const deleteCondition = (i: number) => {
    updateGroup({
      conditions: group.conditions.filter((_, idx) => idx !== i),
    });
  };

  const setConditionField = (i: number, field: keyof MetadataCondition, value: string | number) => {
    const updated = group.conditions.map((c, idx) => {
      if (idx !== i) return c;
      if (field === 'name' && metadataValueTypes[value as string] === 'datetime') {
        return { ...c, name: value as string, value: new Date().getTime() };
      }
      return { ...c, [field]: value };
    });
    updateGroup({ conditions: updated });
  };

  const addSubGroup = () => {
    updateGroup({
      condition_groups: [...group.condition_groups, createEmptyConditionGroup()],
    });
  };

  const deleteSubGroup = (i: number) => {
    updateGroup({
      condition_groups: group.condition_groups.filter((_, idx) => idx !== i),
    });
  };

  const updateSubGroup = (i: number, subGroup: ConditionGroup) => {
    const updated = group.condition_groups.map((g, idx) => (idx === i ? subGroup : g));
    updateGroup({ condition_groups: updated });
  };

  const isNested = depth > 0;

  return (
    <div className={`space-y-2 ${isNested ? 'border rounded-md p-2 bg-muted/30' : ''}`}>
      {/* Header: logical operator + delete */}
      <div className="flex items-center gap-2">
        {isNested && (
          <span className="text-xs text-muted-foreground font-medium">
            {t('knowledgebase.conditionGroup')}
          </span>
        )}
        <RadioGroup
          value={group.logical_operator}
          onValueChange={(value) => updateGroup({ logical_operator: value })}
          className="flex items-center"
        >
          <div className="flex items-center space-x-1">
            {!isNested && (
              <p className="text-muted-foreground text-xs mr-1">
                {t('knowledgebase.logicalOperator')}
              </p>
            )}
            <RadioGroupItem value="and" id={`r-and-${depth}`} />
            <Label htmlFor={`r-and-${depth}`} className="text-xs">AND</Label>
            <RadioGroupItem value="or" id={`r-or-${depth}`} />
            <Label htmlFor={`r-or-${depth}`} className="text-xs">OR</Label>
          </div>
        </RadioGroup>
        {onDelete && (
          <Button variant="outline" size="sm" className="w-5 h-5 p-0 ml-auto" onClick={onDelete}>
            <Trash2Icon className="w-3 h-3" />
          </Button>
        )}
      </div>

      {/* Leaf conditions */}
      <div className="space-y-1">
        {group.conditions.map((condition, i) => (
          <div className="flex items-center space-x-1" key={i}>
            <Select
              value={condition.name}
              onValueChange={(value) => setConditionField(i, 'name', value)}
            >
              <SelectTrigger className="h-6 text-xs w-[120px]">
                <SelectValue placeholder={t('knowledgebase.namePlaceholderShort')} />
              </SelectTrigger>
              <SelectContent className="text-xs">
                <SelectGroup>
                  {metadataConfigs.map((metadata) => (
                    <SelectItem key={metadata.name} value={metadata.name} className="text-xs h-5">
                      {metadata.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <Select
              value={condition.comparison_operator}
              onValueChange={(value) => setConditionField(i, 'comparison_operator', value)}
            >
              <SelectTrigger className="h-6 text-xs w-[80px]">
                <SelectValue placeholder={t('knowledgebase.rulePlaceholderShort')} />
              </SelectTrigger>
              <SelectContent className="w-[80px] text-xs">
                <SelectGroup>
                  {DEFAULT_COMPARATORS.map((op) => (
                    <SelectItem key={op} value={op} className="text-xs h-5">
                      {op}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            {condition.comparison_operator === 'in' || condition.comparison_operator === 'not in' ? (
              <Input
                className="w-32 h-6 text-xs"
                placeholder={t('knowledgebase.inValuePlaceholder')}
                value={condition.value.toString()}
                onChange={(e) => setConditionField(i, 'value', e.target.value)}
              />
            ) : metadataValueTypes[condition.name] === 'datetime' ? (
              <DatetimeInput
                value={(() => {
                  const val = typeof condition.value === 'number'
                    ? condition.value
                    : parseFloat(condition.value as string);
                  return isNaN(val) ? new Date().getTime() : val;
                })()}
                width="sm"
                onValueChange={(value) => setConditionField(i, 'value', value)}
              />
            ) : (
              <Input
                className="w-24 h-6 text-xs"
                value={condition.value.toString()}
                onChange={(e) => setConditionField(i, 'value', e.target.value)}
              />
            )}
            <Button
              variant="outline"
              onClick={() => deleteCondition(i)}
              className="w-5 h-5 p-0"
              size="sm"
            >
              <Trash2Icon className="w-3 h-3" />
            </Button>
          </div>
        ))}
      </div>

      {/* Nested condition groups */}
      {group.condition_groups.map((subGroup, i) => (
        <ConditionGroupEditor
          key={i}
          group={subGroup}
          onChange={(g) => updateSubGroup(i, g)}
          onDelete={() => deleteSubGroup(i)}
          metadataConfigs={metadataConfigs}
          metadataValueTypes={metadataValueTypes}
          depth={depth + 1}
          t={t}
        />
      ))}

      {/* Action buttons */}
      <div className="flex gap-1">
        <Button variant="secondary" onClick={addCondition} className="h-6 text-xs" size="sm">
          <PlusIcon className="w-3 h-3 mr-1" />
          {t('knowledgebase.newFilterRule')}
        </Button>
        {depth < MAX_DEPTH && (
          <Button variant="secondary" onClick={addSubGroup} className="h-6 text-xs" size="sm">
            <PlusIcon className="w-3 h-3 mr-1" />
            {t('knowledgebase.addConditionGroup')}
          </Button>
        )}
      </div>
    </div>
  );
}

/**
 * Convert the UI ConditionGroup tree into the backend MetadataFilteringCondition format.
 * Uses condition_groups for nesting.
 */
export function conditionGroupToPayload(group: ConditionGroup): object | null {
  const hasConditions = group.conditions.length > 0 &&
    group.conditions.some((c) => c.name && c.comparison_operator);
  const hasSubGroups = group.condition_groups.length > 0;

  if (!hasConditions && !hasSubGroups) return null;

  const validConditions = group.conditions
    .filter((c) => c.name && c.comparison_operator)
    .map((c) => {
      if ((c.comparison_operator === 'in' || c.comparison_operator === 'not in') && typeof c.value === 'string') {
        return { ...c, value: c.value.split(',').map((v) => v.trim()).filter(Boolean) };
      }
      return c;
    });

  // If only leaf conditions, no nesting needed - use flat conditions
  if (!hasSubGroups) {
    return {
      logical_operator: group.logical_operator,
      conditions: validConditions,
    };
  }

  // Build condition_groups for each sub-group
  const subPayloads = group.condition_groups
    .map((g) => conditionGroupToPayload(g))
    .filter((p): p is object => p !== null);

  // If we have both leaf conditions and sub-groups, wrap leaf conditions as a sub-group too
  if (hasConditions && validConditions.length > 0) {
    return {
      logical_operator: group.logical_operator,
      condition_groups: [
        { logical_operator: group.logical_operator, conditions: validConditions },
        ...subPayloads,
      ],
    };
  }

  // Only sub-groups
  if (subPayloads.length === 1) {
    // Unwrap single sub-group if same logical operator or only one
    return {
      logical_operator: group.logical_operator,
      condition_groups: subPayloads,
    };
  }

  return {
    logical_operator: group.logical_operator,
    condition_groups: subPayloads,
  };
}
