#lang racket
(define (sorted_list_sum lst)
  (define filtered (filter (lambda (s) (even? (string-length s))) lst))
  (sort filtered (lambda (a b)
                   (cond [(< (string-length a) (string-length b)) #t]
                         [(> (string-length a) (string-length b)) #f]
                         [else (string<? a b)]))))
