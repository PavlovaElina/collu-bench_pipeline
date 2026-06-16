#lang racket
(define (rolling_max numbers)
  (cond [(null? numbers) '()]
        [else (let loop ([nums (cdr numbers)] [cur (car numbers)] [acc (list (car numbers))])
                (cond [(null? nums) (reverse acc)]
                      [else (define m (max cur (car nums)))
                            (loop (cdr nums) m (cons m acc))]))]))
