import { buttonVariants } from '@/components/ui/button';

import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationEllipsis,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination';
import { cn } from '@/lib/utils';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export function PaginationComponent({
  currentPage,
  totalPages,
  onPageChange,
}: PaginationProps) {
  const renderPaginationItems = () => {
    const items = [];
    const maxVisiblePages = 5;

    if (totalPages <= maxVisiblePages) {
      // 显示所有页码
      for (let i = 1; i <= totalPages; i++) {
        items.push(
          <PaginationItem key={i}>
            {i === currentPage ? (
              <PaginationLink
                onClick={() => onPageChange(i)}
                isActive={i === currentPage}
                className={cn(
                  '!shadow-none hover:!text-primary-foreground',
                  buttonVariants({
                    variant: 'default',
                    size: 'icon',
                  }),
                )}
              >
                {i}
              </PaginationLink>
            ) : (
              <PaginationLink
                onClick={() => onPageChange(i)}
                isActive={i === currentPage}
              >
                {i}
              </PaginationLink>
            )}
          </PaginationItem>,
        );
      }
    } else {
      // 显示省略号逻辑（如 1 ... 3 4 5 ... 10）
      const start = Math.max(1, currentPage - 2);
      const end = Math.min(totalPages, currentPage + 2);

      if (start > 1) {
        items.push(
          <PaginationItem key={1}>
            <PaginationLink onClick={() => onPageChange(1)}>1</PaginationLink>
          </PaginationItem>,
        );
        if (start > 2) items.push(<PaginationEllipsis key="start-ellipsis" />);
      }

      for (let i = start; i <= end; i++) {
        items.push(
          <PaginationItem key={i}>
            {i === currentPage ? (
              <PaginationLink
                onClick={() => onPageChange(i)}
                isActive={i === currentPage}
                className={cn(
                  '!shadow-none hover:!text-primary-foreground',
                  buttonVariants({
                    variant: 'default',
                    size: 'icon',
                  }),
                )}
              >
                {i}
              </PaginationLink>
            ) : (
              <PaginationLink
                onClick={() => onPageChange(i)}
                isActive={i === currentPage}
              >
                {i}
              </PaginationLink>
            )}
          </PaginationItem>,
        );
      }

      if (end < totalPages) {
        if (end < totalPages - 1) {
          items.push(<PaginationEllipsis key="end-ellipsis" />);
        }
        items.push(
          <PaginationItem key={totalPages}>
            <PaginationLink onClick={() => onPageChange(totalPages)}>
              {totalPages}
            </PaginationLink>
          </PaginationItem>,
        );
      }
    }

    return items;
  };

  return (
    <Pagination>
      <PaginationContent>
        {/* 首页 */}
        <PaginationItem>
          <PaginationLink
            onClick={() => onPageChange(1)}
            isActive={Boolean(currentPage === 1)}
          >
            首页
          </PaginationLink>
        </PaginationItem>

        {/* 上一页 */}
        <PaginationItem>
          <PaginationPrevious
            onClick={() => onPageChange(currentPage - 1)}
            isActive={currentPage === 1}
          />
          {/* 上一页
                    </PaginationLink> */}
        </PaginationItem>

        {/* 页码 */}
        {renderPaginationItems()}

        {/* 下一页 */}
        <PaginationItem>
          <PaginationNext
            onClick={() => onPageChange(currentPage + 1)}
            isActive={currentPage === totalPages}
          />
          {/* 下一页
                    </PaginationLink> */}
        </PaginationItem>

        {/* 末页 */}
        <PaginationItem>
          <PaginationLink
            onClick={() => onPageChange(totalPages)}
            isActive={currentPage === totalPages}
          >
            末页
          </PaginationLink>
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  );
}
