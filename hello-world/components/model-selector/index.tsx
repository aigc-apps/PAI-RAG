"use client";

import {
  Command,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  ALL_MODEL_NAMES,
  ALL_MODELS,
  OPENAI_MODELS,
  QWEN_MODELS,
} from "@/constants";
import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { Check, ChevronsUpDown } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
export interface ModelConfigurationParams {
  name: string;
  label: string;
}

interface ModelSelectorProps {
  modelName: ALL_MODEL_NAMES;
  setModelName: (name: ALL_MODEL_NAMES) => void;
}

interface CommandModelItemProps {
  model: ModelConfigurationParams;
  handleModelChange: (newModel: ALL_MODEL_NAMES) => Promise<void>;
  selectedModelName: ALL_MODEL_NAMES;
}

function CommandModelItem({
  model,
  handleModelChange,
  selectedModelName,
}: CommandModelItemProps) {
  return (
    <CommandItem
      value={model.name}
      onSelect={handleModelChange}
      className="flex items-center"
    >
      <Check
        className={cn(
          "mr-1 size-4",
          selectedModelName === model.name ? "opacity-100" : "opacity-0",
        )}
      />
      <span className="flex flex-row w-full items-center justify-start gap-2">
        {model.label}
      </span>
    </CommandItem>
  );
}

export default function ModelSelector({
  modelName,
  setModelName,
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false);

  const handleModelChange = useCallback(
    async (newModel: ALL_MODEL_NAMES) => {
      setModelName(newModel);
      setOpen(false);
    },
    [setModelName],
  );

  const selectedModelLabel =
    ALL_MODELS.find((m) => m.name === modelName)?.label || modelName;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        className="min-w-[180px] w-[250px] bg-transparent shadow-none focus:outline-none cursor-pointer hover:bg-gray-100 rounded transition-colors border-none text-gray-600 h-9 px-3 py-2 text-sm focus:ring-1 focus:ring-ring"
        asChild
      >
        <div className="flex items-center pr-2 truncate">
          <span className="flex flex-row items-center justify-start gap-2 text-lg font-semibold">
            {selectedModelLabel}
          </span>
          <ChevronsUpDown className="size-4 opacity-50 ml-auto" />
        </div>
      </PopoverTrigger>
      <PopoverContent className="min-w-[180px] w-[280px] p-0 shadow-md rounded-md">
        <Command>
          <CommandList>
            <CommandGroup heading="OpenAI" className="w-full">
              {OPENAI_MODELS.map((model) => {
                return (
                  <CommandModelItem
                    key={model.name}
                    model={model}
                    handleModelChange={handleModelChange}
                    selectedModelName={modelName}
                  />
                );
              })}
            </CommandGroup>

            <CommandGroup heading="Qwen" className="w-full">
              {QWEN_MODELS.map((model) => {
                return (
                  <CommandModelItem
                    key={model.name}
                    model={model}
                    handleModelChange={handleModelChange}
                    selectedModelName={modelName}
                  />
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
