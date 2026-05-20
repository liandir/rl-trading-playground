import { CandlestickChart } from "lucide-react";
import { PageHeader } from "@/components/shell/PageHeader";
import { EmptyState } from "@/components/ui/empty";

export default function LivePage() {
  return (
    <>
      <PageHeader
        title="Live"
        description="Deploy a trained agent against the Kraken feed and watch it trade in real time."
      />
      <EmptyState
        icon={<CandlestickChart className="h-8 w-8" />}
        title="Coming soon"
        description="Once a checkpoint is selected, this page will subscribe to the Kraken websocket feed and render the agent's actions alongside live OHLC."
      />
    </>
  );
}
